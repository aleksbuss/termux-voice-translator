"""Smoke tests — verify basic module loading and class instantiation.

These tests do NOT call external tools (whisper, Ollama, etc.).
They verify that all code paths can be loaded and objects created
without crashes, ensuring no import errors or initialization bugs.
"""

from __future__ import annotations

import os

import pytest

from src.config import AppConfig, load_config
from src.stt import STTResult, WhisperSTT
from src.translator import LANG_NAMES, OllamaTranslator, TranslationResult
from src.tts import EspeakTTS, PiperTTS, TTSManager, VoiceNotAvailableError
from src.audio import AudioManager
from src.pipeline import PipelineResult, TranslationPipeline


class TestModuleImports:
    """All modules import without errors."""

    def test_config_imports(self):
        from src.config import AppConfig, load_config
        assert callable(load_config)

    def test_stt_imports(self):
        from src.stt import WhisperSTT, STTResult
        assert STTResult is not None

    def test_translator_imports(self):
        from src.translator import OllamaTranslator, TranslationResult, LANG_NAMES
        assert len(LANG_NAMES) >= 15

    def test_tts_imports(self):
        from src.tts import PiperTTS, EspeakTTS, TTSManager, VoiceNotAvailableError
        assert issubclass(VoiceNotAvailableError, Exception)

    def test_audio_imports(self):
        from src.audio import AudioManager
        assert AudioManager is not None

    def test_pipeline_imports(self):
        from src.pipeline import TranslationPipeline, PipelineResult
        assert PipelineResult is not None


class TestDataclasses:
    """Dataclasses instantiate with correct defaults."""

    def test_stt_result(self):
        r = STTResult(text="hello", language="en")
        assert r.text == "hello"
        assert r.duration_sec == 0.0

    def test_translation_result(self):
        r = TranslationResult(text="hallo", source_lang="en", target_lang="de")
        assert r.target_lang == "de"

    def test_pipeline_result(self):
        r = PipelineResult()
        assert r.source_text == ""
        assert r.timings == {}
        assert r.audio_path is None


class TestClassInstantiation:
    """Objects can be created without crashing."""

    def test_whisper_stt_validates_files(self, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh")
        binary.chmod(0o755)
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 10)

        stt = WhisperSTT(str(binary), str(model))
        assert stt is not None

    def test_ollama_translator_default(self):
        tr = OllamaTranslator()
        assert tr._model == "gemma3:4b"

    def test_ollama_translator_custom(self):
        tr = OllamaTranslator(base_url="http://example.com", model="llama3:8b", timeout=30)
        assert tr._model == "llama3:8b"
        assert tr._timeout == 30

    def test_espeak_tts(self):
        tts = EspeakTTS()
        assert tts is not None

    def test_piper_tts(self, tmp_path):
        piper = PiperTTS(str(tmp_path), {"de": "de.onnx"})
        assert piper is not None

    def test_tts_manager_piper_none(self):
        """TTSManager works with piper=None (espeak-only)."""
        mgr = TTSManager(piper=None, espeak=EspeakTTS(), espeak_fallback=True)
        assert mgr is not None

    def test_audio_manager_default(self):
        mgr = AudioManager()
        assert mgr._sample_rate == 16000

    def test_audio_manager_custom(self):
        mgr = AudioManager(sample_rate=8000, max_duration=30)
        assert mgr._max_duration == 30


class TestLangNames:
    """Language name mapping is complete."""

    def test_all_expected_languages_present(self):
        expected = {"ru", "en", "de", "es", "fr", "lv", "it", "pt", "zh", "ja", "ko", "ar", "tr", "uk", "pl"}
        assert expected.issubset(set(LANG_NAMES.keys()))

    def test_values_are_strings(self):
        for code, name in LANG_NAMES.items():
            assert isinstance(code, str) and len(code) == 2
            assert isinstance(name, str) and len(name) > 2


class TestConfigYamlSmoke:
    """Project config.yaml loads and has expected structure."""

    def test_loads_without_error(self):
        config = load_config("config.yaml")
        assert isinstance(config, AppConfig)

    def test_all_sections_present(self):
        config = load_config("config.yaml")
        assert config.stt is not None
        assert config.translation is not None
        assert config.tts is not None
        assert config.audio is not None
        assert config.telegram is not None
        assert config.general is not None
