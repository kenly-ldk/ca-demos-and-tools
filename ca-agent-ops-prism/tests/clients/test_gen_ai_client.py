"""Unit tests for GenAIClient."""

import unittest.mock

from google.genai import types
from prism.server.clients import gen_ai_client
import pydantic
import pytest


class TestGenAIClient:
  """Tests for GenAIClient."""

  @pytest.fixture
  def mock_client(self):
    with unittest.mock.patch(
        "prism.server.clients.gen_ai_client.genai.Client"
    ) as mock:
      yield mock

  @pytest.fixture
  def client(self, mock_client):
    return gen_ai_client.GenAIClient(
        project="test-project", location="us-central1"
    )

  def test_generate_text_success(self, client, mock_client):
    """Tests successful text generation."""
    mock_client_instance = mock_client.return_value
    mock_response = unittest.mock.MagicMock()
    mock_response.text = "Generated Text"
    mock_client_instance.models.generate_content.return_value = mock_response

    result = client.generate_text("prompt")

    assert result == "Generated Text"
    mock_client_instance.models.generate_content.assert_called_with(
        model="gemini-2.5-pro", contents="prompt"
    )

  def test_generate_structured_success(self, client, mock_client):
    """Tests structured generation success."""
    mock_client_instance = mock_client.return_value
    mock_response = unittest.mock.MagicMock()
    mock_response.text = '{"foo": "bar"}'
    mock_client_instance.models.generate_content.return_value = mock_response

    class TestSchema(pydantic.BaseModel):
      foo: str

    result = client.generate_structured("prompt", TestSchema)

    assert result.foo == "bar"
    # Verify generation_config was passed with JSON mime type
    call_args = mock_client_instance.models.generate_content.call_args
    assert call_args
    _, kwargs = call_args
    config = kwargs.get("config")
    assert isinstance(config, types.GenerateContentConfig)
    assert config.response_mime_type == "application/json"

  def test_generate_text_empty_response(self, client, mock_client):
    """Tests handling of empty response."""
    mock_client_instance = mock_client.return_value
    mock_client_instance.models.generate_content.return_value = (
        unittest.mock.MagicMock(text=None)
    )

    result = client.generate_text("prompt")

    assert result is None

  def test_generate_text_exception(self, client, mock_client):
    """Tests exception handling."""
    mock_client_instance = mock_client.return_value
    mock_client_instance.models.generate_content.side_effect = Exception(
        "API Error"
    )

    with pytest.raises(Exception, match="API Error"):
      client.generate_text("prompt")

  def test_ambient_adc_when_no_credentials_file(self, mock_client):
    """No credentials file configured => let the SDK resolve ADC itself."""
    with unittest.mock.patch.object(
        gen_ai_client.settings, "gcp_genai_credentials_file", None
    ):
      client = gen_ai_client.GenAIClient(
          project="test-project", location="us-central1"
      )

    assert client.credentials is None
    _, kwargs = mock_client.call_args
    assert kwargs["credentials"] is None
    assert kwargs["project"] == "test-project"

  def test_explicit_credentials_file(self, mock_client, tmp_path):
    """An explicit credentials file is loaded and handed to the SDK."""
    cred_path = tmp_path / "adc.json"
    cred_path.write_text("{}")
    fake_creds = unittest.mock.MagicMock()

    with unittest.mock.patch.object(
        gen_ai_client.google.auth,
        "load_credentials_from_file",
        return_value=(fake_creds, None),
    ) as mock_load:
      client = gen_ai_client.GenAIClient(
          project="judge-project",
          location="us-central1",
          model="gemini-2.5-flash",
          credentials_file=str(cred_path),
      )

    mock_load.assert_called_once_with(
        str(cred_path),
        scopes=gen_ai_client.GENAI_SCOPES,
        quota_project_id="judge-project",
    )
    assert client.credentials is fake_creds
    assert client.model == "gemini-2.5-flash"
    _, kwargs = mock_client.call_args
    assert kwargs["credentials"] is fake_creds
    assert kwargs["project"] == "judge-project"

  def test_missing_credentials_file_raises(self, mock_client, tmp_path):
    """A configured-but-absent credentials file fails loudly."""
    with pytest.raises(FileNotFoundError):
      gen_ai_client.GenAIClient(
          project="judge-project",
          location="us-central1",
          credentials_file=str(tmp_path / "nope.json"),
      )

  def test_model_falls_back_to_settings(self, mock_client):
    """PRISM_GENAI_CLIENT_MODEL is used when no model argument is given."""
    with unittest.mock.patch.object(
        gen_ai_client.settings, "gcp_genai_model", "gemini-2.5-flash"
    ), unittest.mock.patch.object(
        gen_ai_client.settings, "gcp_genai_credentials_file", None
    ):
      client = gen_ai_client.GenAIClient(
          project="test-project", location="us-central1"
      )

    assert client.model == "gemini-2.5-flash"
