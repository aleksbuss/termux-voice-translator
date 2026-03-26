"""Translation module wrapping Ollama HTTP API.

Uses the /api/generate endpoint for single-turn translation
via a locally running Ollama instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

LANG_NAMES: dict[str, str] = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "lv": "Latvian",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "pl": "Polish",
}


@dataclass
class TranslationResult:
    """Result of a text translation.

    Attributes:
        text: Translated text.
        source_lang: Source language ISO 639-1 code.
        target_lang: Target language ISO 639-1 code.
    """

    text: str
    source_lang: str
    target_lang: str


class OllamaTranslator:
    """Translation via local Ollama LLM.

    Uses ``/api/generate`` (simpler than ``/api/chat`` for single-turn).
    The system prompt strictly constrains output to translation only.

    Args:
        base_url: Ollama API base URL.
        model: Model name to use for translation.
        timeout: Request timeout in seconds.
    """

    SYSTEM_PROMPT = (
        "You are a professional translator. Your ONLY job is to translate text.\n\n"
        "RULES:\n"
        "- Output ONLY the translated text. Nothing else.\n"
        "- Do NOT add explanations, notes, alternatives, or commentary.\n"
        "- Do NOT add quotation marks around the translation.\n"
        '- Do NOT say "Here is the translation" or similar.\n'
        "- Preserve the original tone, style, and register.\n"
        "- If the input is a single word, output a single word.\n"
        "- If the input contains profanity, translate it naturally (do not censor).\n"
        "- If you cannot translate (e.g., gibberish input), output the original text unchanged."
    )

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> TranslationResult:
        """Translate text from source to target language.

        Args:
            text: Source text to translate.
            source_lang: ISO 639-1 code of the source language.
            target_lang: ISO 639-1 code of the target language.

        Returns:
            A ``TranslationResult`` with the translated text.

        Raises:
            ConnectionError: If Ollama is not running.
            TimeoutError: If the request exceeds the timeout.
            ValueError: If the response is empty.
        """
        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)

        prompt = f"Translate from {source_name} to {target_name}:\n\n{text}"

        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": self.SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 500,
            },
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout)
            ) as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                "Cannot connect to Ollama. Is it running?\n"
                "Start it with: ollama serve &"
            ) from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Translation timed out after {self._timeout}s. "
                "The model may be too slow on this device. "
                "Try a smaller model: ollama pull qwen2.5:3b"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}. "
                f"Model: {self._model}. Check that the model is pulled: "
                f"ollama pull {self._model}"
            ) from exc

        try:
            data = resp.json()
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Ollama returned invalid JSON. Response: {resp.text[:200]}"
            ) from exc

        translated = data.get("response", "").strip()

        if not translated:
            raise ValueError(
                "Ollama returned an empty response. "
                f"Model: {self._model}, prompt length: {len(text)} chars"
            )

        logger.info(
            "Translated %d chars (%s->%s) in model=%s",
            len(text), source_lang, target_lang, self._model,
        )
        return TranslationResult(
            text=translated, source_lang=source_lang, target_lang=target_lang
        )

    async def health_check(self) -> bool:
        """Check if Ollama is running and the configured model is available.

        Returns:
            ``True`` if Ollama responds and the model is in the list.
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
            return False

        try:
            data = resp.json()
        except (ValueError, KeyError):
            return False
        models = [m.get("name", "") for m in data.get("models", [])]
        # Ollama model names may include tags like ":latest"
        return any(
            self._model in m or m.startswith(self._model)
            for m in models
        )
