# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recover the compiled BigQuery SQL for Looker Conversational Analytics agents.

The CA API returns a *Looker query* (model / explore / fields / filters / sorts /
limit) for a Looker datasource, never the compiled SQL: Looker itself compiles
LookML -> SQL and runs it internally, so the SQL never travels back in the chat
stream (`DataMessage.generated_sql` stays empty; only `generated_looker_query` is
populated).

This module re-compiles that query spec through the Looker API (create the query,
then fetch `result_format="sql"`, which returns the generated SQL WITHOUT executing
it against BigQuery) and writes the result onto the DataMessage's `generated_sql`
field. Both the trace UI's "Generated SQL" panel and the
`query-contains` assertion read `generated_sql`, so both start working for Looker
agents once this runs.

It mutates the `AskQuestionResponse.response` list of dicts in place — that list is
the single source of truth (the `protobuf_response` property re-parses it on every
access, and the trial trace is re-serialized from it), so writing there reaches
every downstream consumer.

Best-effort and fail-safe by design: it never raises. Any auth/compile failure is
logged and leaves the response untouched (SQL simply stays absent, as before).
"""

import logging
from typing import Any

try:
  import looker_sdk
  from looker_sdk.rtl import api_settings as looker_settings
  from looker_sdk.sdk.api40 import models as looker_models
except Exception:  # pylint: disable=broad-except  # pragma: no cover
  looker_sdk = None
  looker_settings = None
  looker_models = None

logger = logging.getLogger(__name__)


def _looker_instance_uri(agent: Any) -> str | None:
  """Returns the agent's Looker base URL, or None if it is not a Looker agent."""
  config = getattr(agent, "datasource_config", None) or {}
  if not isinstance(config, dict):
    return None
  return config.get("instance_uri") or config.get("looker_instance_uri")


def _make_looker_sdk(base_url: str, client_id: str, client_secret: str):
  """Builds a Looker40SDK from explicit credentials (mirrors AgentService)."""

  class _LookerSettings(looker_settings.ApiSettings):
    """Dynamic settings that ignore looker.ini / env and use passed creds."""

    def read_config(self) -> "looker_settings.SettingsConfig":
      return {
          "base_url": base_url,
          "client_id": client_id,
          "client_secret": client_secret,
          "verify_ssl": "False",
      }

  return looker_sdk.init40(config_settings=_LookerSettings())


def _iter_looker_data_messages(response: Any):
  """Yields the DataMessage dicts that hold a Looker query but no SQL yet.

  Works on the snake_case dicts in `response.response` (built by the client with
  `preserving_proto_field_name=True`), tolerating camelCase just in case.
  """
  items = getattr(response, "response", None) or []
  for item in items:
    if not isinstance(item, dict):
      continue
    sys_msg = item.get("system_message") or item.get("systemMessage")
    if not isinstance(sys_msg, dict):
      continue
    data = sys_msg.get("data")
    if not isinstance(data, dict):
      continue
    if data.get("generated_sql") or data.get("generatedSql"):
      continue  # already have SQL (e.g. a BigQuery message) — leave it
    looker_query = _extract_looker_query(data)
    if not looker_query:
      continue
    if not (looker_query.get("model") and looker_query.get("explore")):
      continue
    yield data, looker_query


def _extract_looker_query(data: dict) -> dict | None:
  """Locates the LookerQuery spec inside a DataMessage dict.

  In practice the CA agent nests it under `data.query.looker` (a Query whose
  `looker` field is the LookerQuery). The proto also has a top-level
  `generated_looker_query`; we accept either. Tolerates camelCase.
  """
  query = data.get("query")
  if isinstance(query, dict):
    looker = query.get("looker")
    if isinstance(looker, dict) and looker.get("model"):
      return looker
  gen = data.get("generated_looker_query") or data.get("generatedLookerQuery")
  if isinstance(gen, dict) and gen.get("model"):
    return gen
  return None


def _write_query_from_looker(looker_query: dict) -> "looker_models.WriteQuery":
  """Maps a CA `LookerQuery` dict to a Looker SDK `WriteQuery`."""
  raw_filters = looker_query.get("filters")
  filters = {}
  if isinstance(raw_filters, dict):
    filters = {k: v for k, v in raw_filters.items() if k}
  elif isinstance(raw_filters, list):
    for f in raw_filters:
      if isinstance(f, dict) and f.get("field"):
        filters[f["field"]] = f.get("value", "")
  limit = looker_query.get("limit")
  return looker_models.WriteQuery(
      model=looker_query["model"],
      view=looker_query["explore"],  # CA "explore" == Looker "view"
      fields=list(looker_query.get("fields") or []) or None,
      filters=filters or None,
      sorts=list(looker_query.get("sorts") or []) or None,
      limit=str(limit) if limit else None,
  )


def recover_generated_sql(response: Any, agent: Any) -> int:
  """Populate `generated_sql` on Looker DataMessages in `response`, in place.

  Args:
      response: an AskQuestionResponse (mutated in place via `.response`).
      agent: the Agent model (for Looker instance URI + client credentials).

  Returns:
      The number of data messages enriched with recovered SQL (0 if not a Looker
      agent, credentials/SDK missing, nothing to recover, or every attempt failed).
  """
  if looker_sdk is None:
    return 0

  base_url = _looker_instance_uri(agent)
  client_id = getattr(agent, "looker_client_id", None)
  client_secret = getattr(agent, "looker_client_secret", None)
  if not (base_url and client_id and client_secret):
    return 0  # not a Looker agent, or missing creds

  # Collect work first so we only authenticate when there is SQL to recover.
  targets = list(_iter_looker_data_messages(response))
  if not targets:
    return 0

  try:
    sdk = _make_looker_sdk(base_url, client_id, client_secret)
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("Looker SQL recovery: SDK init failed: %s", e)
    return 0

  enriched = 0
  for data, looker_query in targets:
    model = looker_query.get("model")
    explore = looker_query.get("explore")
    try:
      created = sdk.create_query(_write_query_from_looker(looker_query))
      # Low-level GET with structure=str: the run/sql endpoint returns the
      # generated SQL as raw text (it does NOT execute against BigQuery). We call
      # get(..., str) rather than run_inline_query() because the SDK's typed
      # deserializer tries to JSON-parse a non-JSON `sql` body on some
      # Python/typing versions; forcing structure=str returns the raw text.
      sql = sdk.get(f"/queries/{created.id}/run/sql", str)
      if isinstance(sql, bytes):
        sql = sql.decode("utf-8", "replace")
      if sql and sql.strip():
        data["generated_sql"] = sql
        enriched += 1
    except Exception as e:  # pylint: disable=broad-except
      logger.warning(
          "Looker SQL recovery failed for %s.%s: %s", model, explore, e
      )

  if enriched:
    logger.info(
        "Looker SQL recovery: enriched %d data message(s) for agent %s",
        enriched,
        getattr(agent, "id", "?"),
    )
  return enriched
