"""Text-to-Speech module for Termux offline voice translator.

Two engines: Piper TTS (primary, high quality ONNX voices) and
espeak-ng (fallback, lower quality but works for any language).
TTSManager orchestrates both with automatic fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceNotAvailableError(Exception):
    """Raised when no TTS voice model is available for the requested language."""


class PiperTTS:
    """Text-to-Speech using Piper (Python library with ONNX Runtime).

    The Piper pre-built binary from GitHub DOES NOT WORK on Android/Termux
    because it targets glibc, while Termux uses Bionic libc.  The Python
    package ``piper-tts`` (via ``pip install piper-tts``) uses ONNX Runtime
    for inference and works correctly.

    Each voice requires TWO files in ``voices_dir``:
    ``<name>.onnx`` and ``<name>.onnx.json`` (config).

    Args:
        voices_dir: Directory containing .onnx voice model files.
        voice_map: Language-to-filename mapping, e.g.
            ``{"de": "de_DE-thorsten-high.onnx", "lv": None}``.
            A ``None`` value means no Piper voice is available for that language.
    """

    _MAX_CHUNK_CHARS = 500

    def __init__(self, voices_dir: str, voice_map: dict[str, str | None]) -> None:
        self._voices_dir = Path(os.path.expandvars(os.path.expanduser(voices_dir)))
        self._voice_map = voice_map
        self._loaded_voices: dict[str, object] = {}

    def _get_voice(self, language: str) -> object:
        """Return a loaded PiperVoice, caching the result."""
        if language in self._loaded_voices:
            return self._loaded_voices[language]

        filename = self._voice_map.get(language)
        if not filename:
            raise VoiceNotAvailableError(
                f"No Piper voice configured for language '{language}'. "
                f"Available: {[k for k, v in self._voice_map.items() if v]}"
            )

        model_path = self._voices_dir / filename
        if not model_path.is_file():
            raise VoiceNotAvailableError(
                f"Piper model file not found: {model_path}. "
                "Download it from https://github.com/rhasspy/piper/blob/master/VOICES.md"
            )

        try:
            from piper import PiperVoice  # type: ignore[import-untyped]
        except ImportError:
            raise VoiceNotAvailableError(
                "piper-tts package is not installed. "
                "Install it with: pip install piper-tts"
            ) from None

        logger.info("Loading Piper voice: %s", model_path)
        voice = PiperVoice.load(str(model_path))
        self._loaded_voices[language] = voice
        return voice

    async def synthesize(self, text: str, language: str, output_path: str) -> str:
        """Convert text to a WAV audio file using Piper.

        For text longer than ``_MAX_CHUNK_CHARS`` the input is split into
        sentences, each synthesized separately, and the resulting WAVs are
        concatenated.

        Args:
            text: Text to speak.
            language: ISO 639-1 code mapped via ``voice_map``.
            output_path: Destination path for the output WAV file.

        Returns:
            The *output_path* on success.

        Raises:
            VoiceNotAvailableError: If no voice is configured/found.
        """
        voice = self._get_voice(language)

        if len(text) > self._MAX_CHUNK_CHARS:
            chunks = _split_sentences(text)
            logger.info("Text is %d chars, split into %d chunks", len(text), len(chunks))
        else:
            chunks = [text]

        loop = asyncio.get_running_loop()

        if len(chunks) == 1:
            await loop.run_in_executor(
                None, self._synthesize_sync, voice, chunks[0], output_path
            )
        else:
            wav_parts: list[str] = []
            try:
                for idx, chunk in enumerate(chunks):
                    part_path = f"{output_path}.part{idx}.wav"
                    await loop.run_in_executor(
                        None, self._synthesize_sync, voice, chunk, part_path
                    )
                    wav_parts.append(part_path)
                _concatenate_wavs(wav_parts, output_path)
            finally:
                for part in wav_parts:
                    try:
                        os.unlink(part)
                    except OSError:
                        pass

        logger.info("Piper TTS wrote %s (%s)", output_path, language)
        return output_path

    @staticmethod
    def _synthesize_sync(voice: object, text: str, output_path: str) -> None:
        """Blocking Piper synthesis — runs in executor."""
        with wave.open(output_path, "wb") as wav_file:
            voice.synthesize(text, wav_file)  # type: ignore[union-attr]


class EspeakTTS:
    """Fallback TTS using espeak-ng.

    Lower quality than Piper but supports virtually every language
    without downloading separate model files.
    """

    _TIMEOUT_SEC = 30

    async def synthesize(self, text: str, language: str, output_path: str) -> str:
        """Synthesize text to a WAV file via espeak-ng.

        Args:
            text: Text to speak.
            language: ISO 639-1 code.
            output_path: Destination WAV path.

        Returns:
            The *output_path* on success.

        Raises:
            RuntimeError: If espeak-ng exits with a non-zero code.
            TimeoutError: If synthesis exceeds 30 seconds.
        """
        proc = await asyncio.create_subprocess_exec(
            "espeak-ng",
            "-v", language,
            "-w", output_path,
            text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"espeak-ng timed out after {self._TIMEOUT_SEC}s"
            ) from None

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"espeak-ng exited with code {proc.returncode}: {err_msg}"
            )

        logger.info("espeak-ng TTS wrote %s (%s)", output_path, language)
        return output_path


class TTSManager:
    """Orchestrates Piper (primary) and espeak-ng (fallback) TTS engines.

    Args:
        piper: A configured PiperTTS instance, or None.
        espeak: An EspeakTTS instance.
        espeak_fallback: Whether to fall back to espeak-ng on failure.
    """

    def __init__(
        self,
        piper: PiperTTS | None,
        espeak: EspeakTTS,
        espeak_fallback: bool = True,
    ) -> None:
        self._piper = piper
        self._espeak = espeak
        self._espeak_fallback = espeak_fallback

    async def synthesize(self, text: str, language: str, output_path: str) -> str:
        """Synthesize speech, trying Piper first with espeak fallback.

        Args:
            text: Text to speak.
            language: ISO 639-1 code.
            output_path: Destination WAV path.

        Returns:
            The *output_path* on success.

        Raises:
            VoiceNotAvailableError: If neither engine can handle the language.
        """
        if self._piper is not None:
            try:
                result = await self._piper.synthesize(text, language, output_path)
                logger.info("TTS engine used: Piper (%s)", language)
                return result
            except VoiceNotAvailableError:
                if not self._espeak_fallback:
                    raise
                logger.warning(
                    "Piper voice not available for '%s', falling back to espeak-ng",
                    language,
                )
            except Exception:
                if not self._espeak_fallback:
                    raise
                logger.warning(
                    "Piper synthesis failed for '%s', falling back to espeak-ng",
                    language, exc_info=True,
                )
        elif not self._espeak_fallback:
            raise VoiceNotAvailableError(
                "Piper TTS is not configured and espeak fallback is disabled"
            )

        result = await self._espeak.synthesize(text, language, output_path)
        logger.info("TTS engine used: espeak-ng (%s)", language)
        return result


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks for TTS."""
    raw = re.split(r"(?<=[.!?;])\s+", text.strip())
    if not raw:
        return [text]

    chunks: list[str] = []
    current = raw[0]
    for fragment in raw[1:]:
        if len(current) + len(fragment) + 1 <= PiperTTS._MAX_CHUNK_CHARS:
            current = f"{current} {fragment}"
        else:
            chunks.append(current)
            current = fragment
    chunks.append(current)
    return chunks


def _concatenate_wavs(parts: list[str], output_path: str) -> None:
    """Concatenate multiple WAV files (same format) into one."""
    if not parts:
        raise ValueError("No WAV parts to concatenate")

    with wave.open(parts[0], "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for part_path in parts[1:]:
        with wave.open(part_path, "rb") as wf:
            frames.append(wf.readframes(wf.getnframes()))

    with wave.open(output_path, "wb") as out:
        out.setparams(params)
        for frame_data in frames:
            out.writeframes(frame_data)
