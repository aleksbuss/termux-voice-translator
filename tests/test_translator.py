"""Tests for src.translator (OllamaTranslator)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.translator import OllamaTranslator, TranslationResult


@pytest.fixture
def translator():
    return OllamaTranslator(base_url="http://localhost:11434", model="gemma3:4b")


@pytest.mark.asyncio
@patch("src.translator.httpx.AsyncClient")
async def test_translate_constructs_correct_request(mock_client_cls, translator):
    """Verify the JSON body sent to Ollama."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Hallo"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    await translator.translate("Hello", "en", "de")

    call_kwargs = mock_client.post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["model"] == "gemma3:4b"
    assert "Translate from English to German" in body["prompt"]
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.3


@pytest.mark.asyncio
@patch("src.translator.httpx.AsyncClient")
async def test_translate_parses_response(mock_client_cls, translator):
    """Verify translated text is extracted from response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Hallo, wie geht es dir?"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    result = await translator.translate("Hello, how are you?", "en", "de")

    assert result.text == "Hallo, wie geht es dir?"
    assert isinstance(result, TranslationResult)


@pytest.mark.asyncio
@patch("src.translator.httpx.AsyncClient")
async def test_translate_strips_whitespace(mock_client_cls, translator):
    """Response with extra whitespace should be stripped."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "  Hallo  \n\n"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    result = await translator.translate("Hi", "en", "de")
    assert result.text == "Hallo"


@pytest.mark.asyncio
@patch("src.translator.httpx.AsyncClient")
async def test_translate_raises_on_connection_error(mock_client_cls, translator):
    """ConnectionError with actionable message when Ollama not running."""
    import httpx

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    with pytest.raises(ConnectionError, match="ollama serve"):
        await translator.translate("Hello", "en", "de")


@pytest.mark.asyncio
@patch("src.translator.httpx.AsyncClient")
async def test_translate_raises_on_empty_response(mock_client_cls, translator):
    """ValueError when Ollama returns empty response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": ""}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    with pytest.raises(ValueError, match="empty response"):
        await translator.translate("Hello", "en", "de")


@pytest.mark.asyncio
@patch("src.translator.httpx.AsyncClient")
async def test_health_check_returns_true(mock_client_cls, translator):
    """health_check returns True when model is in the list."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"name": "gemma3:4b"}, {"name": "llama3.2:3b"}]
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    assert await translator.health_check() is True


@pytest.mark.asyncio
@patch("src.translator.httpx.AsyncClient")
async def test_health_check_returns_false(mock_client_cls, translator):
    """health_check returns False when model not in list."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "models": [{"name": "llama3.2:3b"}]
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client_cls.return_value = mock_client

    assert await translator.health_check() is False
