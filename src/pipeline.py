"""Main translation pipeline orchestrating the full voice-to-voice flow.

Chains: audio input -> STT -> translation -> TTS -> audio playback.
Each step is timed independently for performance monitoring.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.audio import AudioManager
from src.stt import STTResult, WhisperSTT
from src.translator import LANG_NAMES, OllamaTranslator, TranslationResult
from src.tts import TTSManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result returned after a full or partial translation pipeline run.

    Attributes:
        source_text: Transcribed or user-provided source text.
        source_language: Detected or specified ISO 639-1 language code.
        translated_text: Text after translation.
        target_language: Target ISO 639-1 language code.
        audio_path: Path to the generated TTS audio file, if any.
        timings: Timing breakdown in milliseconds.
    """

    source_text: str = ""
    source_language: str = ""
    translated_text: str = ""
    target_language: str = ""
    audio_path: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


class TranslationPipeline:
    """Orchestrates the full voice-to-voice translation pipeline.

    Args:
        stt: Speech-to-text engine (whisper.cpp wrapper).
        translator: Translation engine (Ollama wrapper).
        tts: Text-to-speech manager (Piper + espeak fallback).
        audio: Audio utilities (recording, conversion, playback).
        show_source: Print recognised source text to the terminal.
        show_translated: Print translated text to the terminal.
    """

    def __init__(
        self,
        stt: WhisperSTT,
        translator: OllamaTranslator,
        tts: TTSManager,
        audio: AudioManager,
        show_source: bool = True,
        show_translated: bool = True,
    ) -> None:
        self._stt = stt
        self._translator = translator
        self._tts = tts
        self._audio = audio
        self._show_source = show_source
        self._show_translated = show_translated

    async def translate_voice(
        self,
        audio_input: str,
        source_lang: str,
        target_lang: str,
        play_result: bool = True,
        save_audio: str | None = None,
    ) -> PipelineResult:
        """Run the full voice-to-voice translation pipeline.

        Args:
            audio_input: Path to audio file or ``"mic"`` for live recording.
            source_lang: ISO 639-1 code or ``"auto"``.
            target_lang: ISO 639-1 target language code.
            play_result: Whether to play the translated audio.
            save_audio: Optional path to persist the translated audio.

        Returns:
            A ``PipelineResult`` with all data and timing breakdown.
        """
        total_start = time.monotonic()
        tmp_files: list[str] = []
        result = PipelineResult(target_language=target_lang)

        try:
            # Recording (optional)
            if audio_input == "mic":
                rec_path = self._make_tmp("rec_", ".wav")
                tmp_files.append(rec_path)
                await self._audio.record(rec_path)
                audio_input = rec_path

            # STT
            print("Transcribing with Whisper...")
            stt_start = time.monotonic()
            stt_result: STTResult = await self._stt.transcribe(
                audio_input, language=source_lang,
            )
            result.timings["stt_ms"] = self._elapsed_ms(stt_start)

            result.source_text = stt_result.text.strip()
            result.source_language = stt_result.language

            if self._show_source:
                print(f'Source ({result.source_language}): "{result.source_text}"')

            if not result.source_text:
                print("No speech detected.")
                result.timings["translate_ms"] = 0.0
                result.timings["tts_ms"] = 0.0
                result.timings["total_ms"] = self._elapsed_ms(total_start)
                return result

            # Translate + TTS + Play
            await self._translate_and_speak(
                result, target_lang=target_lang,
                play_result=play_result, save_audio=save_audio,
                tmp_files=tmp_files,
            )

            result.timings["total_ms"] = self._elapsed_ms(total_start)
            self._print_timings(result.timings)
            return result

        except Exception:
            logger.exception("Pipeline error during translate_voice")
            raise
        finally:
            self._cleanup(tmp_files)

    async def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        play_result: bool = True,
        save_audio: str | None = None,
    ) -> PipelineResult:
        """Translate plain text, skipping the STT stage.

        Args:
            text: Source text to translate.
            source_lang: ISO 639-1 source language code.
            target_lang: ISO 639-1 target language code.
            play_result: Whether to play the translated audio.
            save_audio: Optional path to persist the translated audio.

        Returns:
            A ``PipelineResult`` (``stt_ms`` will be ``0.0``).
        """
        total_start = time.monotonic()
        tmp_files: list[str] = []
        result = PipelineResult(
            source_text=text,
            source_language=source_lang,
            target_language=target_lang,
        )
        result.timings["stt_ms"] = 0.0

        try:
            if self._show_source:
                print(f'Source ({source_lang}): "{text}"')

            await self._translate_and_speak(
                result, target_lang=target_lang,
                play_result=play_result, save_audio=save_audio,
                tmp_files=tmp_files,
            )

            result.timings["total_ms"] = self._elapsed_ms(total_start)
            self._print_timings(result.timings)
            return result

        except Exception:
            logger.exception("Pipeline error during translate_text")
            raise
        finally:
            self._cleanup(tmp_files)

    async def _translate_and_speak(
        self,
        result: PipelineResult,
        *,
        target_lang: str,
        play_result: bool,
        save_audio: str | None,
        tmp_files: list[str],
    ) -> None:
        """Shared tail: translate -> TTS -> play."""
        target_name = LANG_NAMES.get(target_lang, target_lang)

        # Translation
        print(f"Translating to {target_name}...")
        tr_start = time.monotonic()
        tr_result: TranslationResult = await self._translator.translate(
            result.source_text,
            source_lang=result.source_language,
            target_lang=target_lang,
        )
        result.timings["translate_ms"] = self._elapsed_ms(tr_start)
        result.translated_text = tr_result.text.strip()

        if self._show_translated:
            print(f'Translation ({target_lang}): "{result.translated_text}"')

        # TTS
        print("Generating speech...")
        tts_start = time.monotonic()
        tts_path = self._make_tmp("tts_", ".wav")
        tmp_files.append(tts_path)
        await self._tts.synthesize(
            result.translated_text, language=target_lang, output_path=tts_path,
        )
        result.timings["tts_ms"] = self._elapsed_ms(tts_start)
        result.audio_path = tts_path

        # Save
        if save_audio:
            shutil.copy2(tts_path, save_audio)
            result.audio_path = save_audio

        # Playback (use tts_path — it's always valid; save_audio may be external)
        if play_result:
            print("Playing translation...")
            await self._audio.play(result.audio_path)

    @staticmethod
    def _make_tmp(prefix: str, suffix: str) -> str:
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        return path

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.monotonic() - start) * 1000, 1)

    @staticmethod
    def _print_timings(timings: dict[str, float]) -> None:
        def _sec(key: str) -> str:
            return f"{timings.get(key, 0) / 1000:.1f}s"

        print(
            f"Done! (STT: {_sec('stt_ms')} | "
            f"Translate: {_sec('translate_ms')} | "
            f"TTS: {_sec('tts_ms')} | "
            f"Total: {_sec('total_ms')})"
        )

    @staticmethod
    def _cleanup(paths: list[str]) -> None:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                logger.debug("Failed to remove temp file: %s", p)
