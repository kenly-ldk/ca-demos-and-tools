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

"""Client for interacting with the Google Vertex AI API."""

import logging
import os
from typing import Any, Type, TypeVar

from google import genai
from google.genai import types
import google.auth
import pydantic

from prism.server.config import settings

# Default model configuration
DEFAULT_MODEL = "gemini-2.5-pro"

# Scope required to call the Vertex AI Gen AI endpoints.
GENAI_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

ResponseSchema = TypeVar("ResponseSchema", bound=pydantic.BaseModel)


class GenAIClient:
  """A client for interacting with the Google Gen AI API.

  This client uses the `google-genai` SDK (google.genai).
  """

  def __init__(
      self,
      project: str,
      location: str,
      model: str | None = None,
      credentials_file: str | None = None,
  ):
    """Initializes the GenAIClient.

    Args:
        project: GCP project ID.
        location: Location for Vertex AI.
        model: The name of the Gemini model to use. If not specified, falls back
          to PRISM_GENAI_CLIENT_MODEL, then to the default model.
        credentials_file: Path to a service-account key or gcloud ADC JSON file
          to authenticate the Gen AI calls with. If not specified, falls back to
          PRISM_GENAI_CLIENT_CREDENTIALS_FILE; if that is empty too, the
          process-wide Application Default Credentials are used (the historical
          behaviour). Supplying this lets the LLM judge run under a different
          identity/project than the rest of PRISM (GDA agents, BigQuery).
    """
    try:
      self.project = project
      self.location = location

      if not model:
        model = settings.gcp_genai_model
      if not model:
        logging.info(
            "[GenAI] No model specified, using default model: %s",
            DEFAULT_MODEL,
        )
        model = DEFAULT_MODEL
      self.model = model

      if not credentials_file:
        credentials_file = settings.gcp_genai_credentials_file
      self.credentials = self._load_credentials(credentials_file)

      # Use the new Google Gen AI SDK
      self.client = genai.Client(
          vertexai=True,
          project=self.project,
          location=self.location,
          credentials=self.credentials,
      )

    except Exception as e:  # pylint: disable=broad-except
      logging.error("[GenAI] Failed to initialize Gen AI Client: %s", e)
      raise

  def _load_credentials(
      self, credentials_file: str | None
  ) -> google.auth.credentials.Credentials | None:
    """Loads explicit credentials from a file, or None for ambient ADC.

    Args:
        credentials_file: Path to a credentials JSON file, or falsy to use the
          process-wide Application Default Credentials.

    Returns:
        The loaded credentials, or None to let the SDK resolve ADC itself.
    """
    if not credentials_file:
      logging.info(
          "[GenAI] No credentials file configured; using ambient ADC for"
          " project %s",
          self.project,
      )
      return None

    path = os.path.expanduser(os.path.expandvars(credentials_file))
    if not os.path.isfile(path):
      raise FileNotFoundError(
          f"[GenAI] Credentials file not found: {path} (from"
          " PRISM_GENAI_CLIENT_CREDENTIALS_FILE or the credentials_file"
          " argument)"
      )

    credentials, _ = google.auth.load_credentials_from_file(
        path, scopes=GENAI_SCOPES, quota_project_id=self.project
    )
    logging.info(
        "[GenAI] Using explicit credentials from %s (quota project: %s)",
        path,
        self.project,
    )
    return credentials

  def generate_text(
      self,
      prompt: str,
  ) -> str | None:
    """Generates text using the specified Gemini model.

    Args:
        prompt: The text prompt to send to the model.

    Returns:
        The generated text as a string, or None if failed.
    """
    try:
      response = self.client.models.generate_content(
          model=self.model,
          contents=prompt,
      )

      if response and response.text:
        return response.text

      logging.warning("Gen AI response was empty or malformed.")
      return None

    except Exception as e:  # pylint: disable=broad-except
      logging.error(
          "[GenAI] Error during text generation: %s", e, exc_info=True
      )
      raise

  def generate_structured(
      self,
      prompt: str,
      response_schema: Type[ResponseSchema],
  ) -> ResponseSchema | None:
    """Generates a structured response from the LLM with the given schema.

    Args:
        prompt: The text prompt to send to the model.
        response_schema: The Pydantic model class to use for validation.

    Returns:
        An instance of the response_schema or None if generation failed.
    """
    try:
      # Clean schema to be compatible with Gen AI (no 'const')
      schema_dict = response_schema.model_json_schema()
      cleaned_schema = self._clean_schema(schema_dict)

      config = types.GenerateContentConfig(
          response_mime_type="application/json",
          response_schema=cleaned_schema,
      )

      response = self.client.models.generate_content(
          model=self.model,
          contents=prompt,
          config=config,
      )

      if response and response.text:
        return response_schema.model_validate_json(response.text)

      logging.warning("Gen AI structured response was empty.")
      return None

    except Exception as e:  # pylint: disable=broad-except
      logging.error(
          "[GenAI] Error during structured generation: %s",
          e,
          exc_info=True,
      )
      raise

  def _clean_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively cleans schema to be compatible with Vertex AI.

    - Removes 'const' keys, replacing with 'enum'.
    - Removes 'null' types from 'anyOf' (Vertex AI doesn't support NULL type).
    - Hoists single-item 'anyOf' to the parent level.

    Args:
        schema: The JSON schema dictionary to clean.

    Returns:
        The cleaned JSON schema dictionary.
    """
    if not isinstance(schema, dict):
      return schema

    # Create a copy to modify
    clean = schema.copy()

    # Replace 'const' with 'enum'
    if "const" in clean:
      clean["enum"] = [clean.pop("const")]

    # Handle 'oneOf' or 'anyOf'
    for key in ["oneOf", "anyOf"]:
      if key in clean and isinstance(clean[key], list):
        # Filter out null types
        items = [
            item
            for item in clean[key]
            if not (isinstance(item, dict) and item.get("type") == "null")
        ]

        if not items:
          clean.pop(key)
        elif len(items) == 1:
          # Hoist the single item
          single_item = items[0]
          clean.pop(key)
          if isinstance(single_item, dict):
            clean.update(single_item)
        else:
          # Keep items but ensure we use anyOf (Vertex AI preference)
          clean.pop(key)
          clean["anyOf"] = items

    # Remove 'discriminator' (Vertex AI doesn't support it)
    if "discriminator" in clean:
      clean.pop("discriminator")

    # Recurse
    for key, value in clean.items():
      if isinstance(value, dict):
        clean[key] = self._clean_schema(value)
      elif isinstance(value, list):
        clean[key] = [
            self._clean_schema(item) if isinstance(item, dict) else item
            for item in value
        ]

    return clean
