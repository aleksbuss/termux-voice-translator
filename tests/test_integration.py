"""Integration tests — test full module interactions with mocks.

These tests verify that modules work together correctly:
- Config → Pipeline init
- Pipeline with mocked external tools
- CLI argument parsing
- End-to-end flow with all components mocked
"""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig, load_config
from src.stt import STTResult
from src.translator import TranslationResult
from src.pipeline import TranslationPipeline, PipelineResult
from src.main import create_parser, init_pipeline


class TestCLIParser:
    """Argument parser produces correct values."""

    def test_defaults(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.source_lang is None
        assert args.target_lang is None
        assert args.file is None
        assert args.text is None
        assert args.continuous is False
        assert args.telegram is False
        assert args.no_play is False

    def test_from_to(self):
        parser = create_parser()
        args = parser.parse_args(["--from", "ru", "--to", "en"])
        assert args.source_lang == "ru"
        assert args.target_lang == "en"

    def test_file_input(self):
        parser = create_parser()
        args = parser.parse_args(["--file", "test.ogg", "--to", "de"])
        assert args.file == "test.ogg"
        assert args.target_lang == "de"

    def test_text_input(self):
        parser = create_parser()
        args = parser.parse_args(["--text", "Hello world"])
        assert args.text == "Hello world"

    def test_continuous_mode(self):
        parser = create_parser()
        args = parser.parse_args(["--continuous", "--to", "en"])
        assert args.continuous is True

    def test_short_flags(self):
        parser = create_parser()
        args = parser.parse_args(["-f", "test.ogg", "-t", "hello", "-c", "-s", "out.wav"])
        assert args.file == "test.ogg"
        assert args.text == "hello"
        assert args.continuous is True
        assert args.save == "out.wav"

    def test_no_play_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--no-play"])
        assert args.no_play is True

    def test_config_path(self):
        parser = create_parser()
        args = parser.parse_args(["--config", "/custom/config.yaml"])
        assert args.config == "/custom/config.yaml"


class TestInitPipeline:
    """Pipeline initialization from config."""

    def test_init_pipeline_with_valid_config(self, tmp_path):
        """Pipeline can be created when binary and model exist."""
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh")
        binary.chmod(0o755)
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 10)
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()

        config = AppConfig()
        config.stt.whisper_binary = str(binary)
        config.stt.whisper_model = str(model)
        config.tts.piper_voices_dir = str(voices_dir)

        pipeline = init_pipeline(config)
        assert isinstance(pipeline, TranslationPipeline)

    def test_init_pipeline_missing_binary(self):
        """Pipeline init fails with clear error when binary missing."""
        config = AppConfig()
        config.stt.whisper_binary = "/nonexistent/whisper-cli"

        with pytest.raises(FileNotFoundError, match="whisper-cli"):
            init_pipeline(config)


class TestEndToEndMocked:
    """Full pipeline flow with all externals mocked."""

    @pytest.mark.asyncio
    async def test_voice_translation_flow(self, tmp_path):
        """Full voice -> text -> translate -> TTS -> audio flow."""
        stt = AsyncMock()
        stt.transcribe = AsyncMock(
            return_value=STTResult(text="Привет мир", language="ru")
        )

        translator = AsyncMock()
        translator.translate = AsyncMock(
            return_value=TranslationResult(
                text="Hallo Welt", source_lang="ru", target_lang="de"
            )
        )

        tts_output = str(tmp_path / "tts_output.wav")
        tts = AsyncMock()
        tts.synthesize = AsyncMock(return_value=tts_output)

        audio = AsyncMock()
        audio.play = AsyncMock()

        pipeline = TranslationPipeline(
            stt=stt, translator=translator, tts=tts, audio=audio,
            show_source=False, show_translated=False,
        )

        audio_file = tmp_path / "input.wav"
        audio_file.write_bytes(b"\x00" * 1000)

        result = await pipeline.translate_voice(
            str(audio_file), "auto", "de", play_result=True
        )

        # Verify full chain executed
        assert result.source_text == "Привет мир"
        assert result.source_language == "ru"
        assert result.translated_text == "Hallo Welt"
        assert result.target_language == "de"
        assert result.timings["stt_ms"] >= 0
        assert result.timings["translate_ms"] >= 0
        assert result.timings["tts_ms"] >= 0

        # Verify call order
        stt.transcribe.assert_called_once()
        translator.translate.assert_called_once_with(
            "Привет мир", source_lang="ru", target_lang="de"
        )
        tts.synthesize.assert_called_once()
        audio.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_translation_flow(self, tmp_path):
        """Text-only mode skips STT entirely."""
        stt = AsyncMock()
        translator = AsyncMock()
        translator.translate = AsyncMock(
            return_value=TranslationResult(
                text="Hello world", source_lang="ru", target_lang="en"
            )
        )

        tts = AsyncMock()
        tts.synthesize = AsyncMock(return_value="/tmp/out.wav")
        audio = AsyncMock()

        pipeline = TranslationPipeline(
            stt=stt, translator=translator, tts=tts, audio=audio,
            show_source=False, show_translated=False,
        )

        result = await pipeline.translate_text(
            "Привет мир", "ru", "en", play_result=False
        )

        assert result.translated_text == "Hello world"
        assert result.timings["stt_ms"] == 0.0
        stt.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_mic_recording_flow(self):
        """Mic mode calls audio.record first."""
        stt = AsyncMock()
        stt.transcribe = AsyncMock(
            return_value=STTResult(text="test", language="en")
        )
        translator = AsyncMock()
        translator.translate = AsyncMock(
            return_value=TranslationResult(text="тест", source_lang="en", target_lang="ru")
        )
        tts = AsyncMock()
        tts.synthesize = AsyncMock(return_value="/tmp/out.wav")
        audio = AsyncMock()
        audio.record = AsyncMock(return_value="/tmp/rec.wav")

        pipeline = TranslationPipeline(
            stt=stt, translator=translator, tts=tts, audio=audio,
            show_source=False, show_translated=False,
        )

        result = await pipeline.translate_voice(
            "mic", "auto", "ru", play_result=False
        )

        audio.record.assert_called_once()
        assert result.translated_text == "тест"

    @pytest.mark.asyncio
    async def test_save_audio_copies_file(self, tmp_path):
        """--save flag copies TTS output to specified path."""
        stt = AsyncMock()
        translator = AsyncMock()
        translator.translate = AsyncMock(
            return_value=TranslationResult(text="hi", source_lang="ru", target_lang="en")
        )
        tts = AsyncMock()
        tts.synthesize = AsyncMock(return_value="/tmp/out.wav")
        audio = AsyncMock()

        pipeline = TranslationPipeline(
            stt=stt, translator=translator, tts=tts, audio=audio,
            show_source=False, show_translated=False,
        )

        save_path = str(tmp_path / "saved.wav")

        with patch("src.pipeline.shutil.copy2") as mock_copy:
            result = await pipeline.translate_text(
                "Привет", "ru", "en", play_result=False, save_audio=save_path
            )

            mock_copy.assert_called_once()
            assert result.audio_path == save_path
