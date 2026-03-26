"""Tests for src.pipeline (TranslationPipeline)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.stt import STTResult
from src.translator import TranslationResult
from src.pipeline import TranslationPipeline, PipelineResult


def _make_pipeline(
    stt_text: str = "Hello",
    stt_lang: str = "en",
    translated: str = "Hallo",
) -> tuple[TranslationPipeline, dict]:
    """Create a pipeline with mocked components."""
    stt = AsyncMock()
    stt.transcribe = AsyncMock(
        return_value=STTResult(text=stt_text, language=stt_lang)
    )

    translator = AsyncMock()
    translator.translate = AsyncMock(
        return_value=TranslationResult(
            text=translated, source_lang=stt_lang, target_lang="de"
        )
    )

    tts = AsyncMock()
    tts.synthesize = AsyncMock(return_value="/tmp/tts_out.wav")

    audio = AsyncMock()
    audio.record = AsyncMock(return_value="/tmp/rec.wav")
    audio.play = AsyncMock()

    pipeline = TranslationPipeline(
        stt=stt, translator=translator, tts=tts, audio=audio,
        show_source=False, show_translated=False,
    )

    return pipeline, {"stt": stt, "translator": translator, "tts": tts, "audio": audio}


@pytest.mark.asyncio
async def test_full_pipeline_voice(tmp_path):
    """All steps called in order, PipelineResult has all fields."""
    pipeline, mocks = _make_pipeline()

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    result = await pipeline.translate_voice(
        str(audio_file), "auto", "de", play_result=False
    )

    assert isinstance(result, PipelineResult)
    assert result.source_text == "Hello"
    assert result.source_language == "en"
    assert result.translated_text == "Hallo"
    assert result.target_language == "de"
    mocks["stt"].transcribe.assert_called_once()
    mocks["translator"].translate.assert_called_once()
    mocks["tts"].synthesize.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_skips_on_empty_stt(tmp_path):
    """Translator and TTS not called when STT returns empty text."""
    pipeline, mocks = _make_pipeline(stt_text="", stt_lang="unknown")

    audio_file = tmp_path / "silence.wav"
    audio_file.write_bytes(b"\x00" * 100)

    result = await pipeline.translate_voice(
        str(audio_file), "auto", "de", play_result=False
    )

    assert result.source_text == ""
    mocks["translator"].translate.assert_not_called()
    mocks["tts"].synthesize.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_translate_text_skips_stt():
    """translate_text does not call STT."""
    pipeline, mocks = _make_pipeline()

    result = await pipeline.translate_text(
        "Hello", "en", "de", play_result=False
    )

    assert result.source_text == "Hello"
    assert result.translated_text == "Hallo"
    assert result.timings["stt_ms"] == 0.0
    mocks["stt"].transcribe.assert_not_called()
    mocks["translator"].translate.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_measures_timing(tmp_path):
    """Timings dict has required keys with non-negative values."""
    pipeline, _ = _make_pipeline()

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"\x00" * 100)

    result = await pipeline.translate_voice(
        str(audio_file), "auto", "de", play_result=False
    )

    assert "stt_ms" in result.timings
    assert "translate_ms" in result.timings
    assert "tts_ms" in result.timings
    assert "total_ms" in result.timings
    assert all(v >= 0 for v in result.timings.values())
