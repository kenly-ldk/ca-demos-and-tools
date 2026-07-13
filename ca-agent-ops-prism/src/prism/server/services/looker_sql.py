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

The CA API returns a *Looker query* (nested at a DataMessage's `query.looker`:
model / explore / fields / filters / sorts / limit) for a Looker datasource, never
the compiled SQL: Looker itself compiles LookML -> SQL and runs it internally, so
the SQL never travels back in the chat stream (`DataMessage.generated_sql` stays
empty).

This module re-compiles that query spec through the Looker API (create the query,
then fetch `result_format="sql"`, which returns the generated SQL WITHOUT executing
it against BigQuery) and inserts the result as a NEW `generated_sql` DataMessage
right after the Looker query message. The Looker query message is left intact, so
the trace shows BOTH — a "Data Query" panel (the Looker query) AND a "Generated
SQL" panel — mirroring how a BigQuery agent already emits separate query and
generated_sql messages. Both the trace UI and the `query-contains` assertion read
`generated_sql`, so both start working for Looker agents once this runs.

It mutates the `AskQuestionResponse.response` list of dicts in place — that list is
the single source of truth (the `protobuf_response` property re-parses it on every
access, and the trial trace is re-serialized from it), so the inserted message
reaches every downstream consumer.

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


def _data_dict(item: Any) -> dict | None:
  """Returns a message's system_message.data dict, or None."""
  if not isinstance(item, dict):
    return None
  sys_msg = item.get("system_message") or item.get("systemMessage")
  if not isinstance(sys_msg, dict):
    return None
  data = sys_msg.get("data")
  return data if isinstance(data, dict) else None


def _iter_looker_query_messages(items: list):
  """Yields (index, looker_query) for each DataMessage holding a Looker query.

  Works on the snake_case dicts in `response.response` (built by the client with
  `preserving_proto_field_name=True`), tolerating camelCase just in case.
  """
  for idx, item in enumerate(items):
    data = _data_dict(item)
    if data is None:
      continue
    looker_query = _extract_looker_query(data)
    if not looker_query:
      continue
    if not (looker_query.get("model") and looker_query.get("explore")):
      continue
    yield idx, looker_query


def _make_generated_sql_message(source_item: Any, sql: str) -> dict:
  """A synthetic DataMessage carrying only the recovered SQL.

  Inserting this as its OWN message (rather than adding `generated_sql` to the
  Looker query's message) mirrors how a BigQuery agent already lays out its trace
  — separate `query` and `generated_sql` messages — so PRISM's timeline renders
  BOTH the "Data Query" (the Looker query) and a "Generated SQL" panel. It also
  keeps the original Looker query message untouched.
  """
  msg: dict = {"system_message": {"data": {"generated_sql": sql}}}
  if isinstance(source_item, dict) and source_item.get("timestamp"):
    msg["timestamp"] = source_item["timestamp"]
  return msg


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
  """Add a recovered-SQL DataMessage after each Looker query, in place.

  For each Looker query message, re-compiles the query to SQL via the Looker API
  and inserts a NEW `generated_sql` DataMessage right after it (leaving the Looker
  query message intact), so the trace shows BOTH. Mutates `response.response`.

  Args:
      response: an AskQuestionResponse (mutated in place via `.response`).
      agent: the Agent model (for Looker instance URI + client credentials).

  Returns:
      The number of recovered-SQL messages inserted (0 if not a Looker agent,
      credentials/SDK missing, nothing to recover, or every attempt failed).
  """
  if looker_sdk is None:
    return 0

  base_url = _looker_instance_uri(agent)
  client_id = getattr(agent, "looker_client_id", None)
  client_secret = getattr(agent, "looker_client_secret", None)
  if not (base_url and client_id and client_secret):
    return 0  # not a Looker agent, or missing creds

  items = getattr(response, "response", None)
  if not isinstance(items, list):
    return 0

  # If any data message already carries SQL, do nothing: either a BigQuery agent
  # (SQL is native) or a Looker response we already enriched (idempotent).
  for item in items:
    data = _data_dict(item)
    if data and (data.get("generated_sql") or data.get("generatedSql")):
      return 0

  # Collect work first so we only authenticate when there is SQL to recover.
  targets = list(_iter_looker_query_messages(items))
  if not targets:
    return 0

  try:
    sdk = _make_looker_sdk(base_url, client_id, client_secret)
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("Looker SQL recovery: SDK init failed: %s", e)
    return 0

  # Recover SQL per query message (index -> sql), then insert the synthetic
  # messages from the highest index down so earlier indices stay valid.
  recovered: list[tuple[int, str]] = []
  for idx, looker_query in targets:
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
        recovered.append((idx, sql))
    except Exception as e:  # pylint: disable=broad-except
      logger.warning(
          "Looker SQL recovery failed for %s.%s: %s", model, explore, e
      )

  for idx, sql in sorted(recovered, key=lambda r: r[0], reverse=True):
    items.insert(idx + 1, _make_generated_sql_message(items[idx], sql))

  enriched = len(recovered)
  if enriched:
    logger.info(
        "Looker SQL recovery: enriched %d data message(s) for agent %s",
        enriched,
        getattr(agent, "id", "?"),
    )
  return enriched
