"""Tests for src.config — configuration loading and defaults."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from src.config import (
    AppConfig,
    AudioConfig,
    GeneralConfig,
    STTConfig,
    TTSConfig,
    TelegramConfig,
    TranslationConfig,
    load_config,
)


class TestDefaults:
    """Verify all defaults are sane without any config file."""

    def test_load_missing_file_returns_defaults(self):
        config = load_config("/nonexistent/config.yaml")
        assert isinstance(config, AppConfig)
        assert config.stt.engine == "whisper-cpp"
        assert config.translation.model == "gemma3:4b"
        assert config.general.default_target == "de"

    def test_stt_defaults(self):
        stt = STTConfig()
        assert stt.language == "auto"
        assert "whisper-cli" in stt.whisper_binary

    def test_translation_defaults(self):
        tr = TranslationConfig()
        assert tr.ollama_url == "http://localhost:11434"
        assert tr.timeout == 120

    def test_tts_defaults(self):
        tts = TTSConfig()
        assert tts.engine == "piper"
        assert tts.espeak_fallback is True
        assert "de" in tts.voices

    def test_audio_defaults(self):
        audio = AudioConfig()
        assert audio.sample_rate == 16000
        assert audio.max_duration == 60

    def test_telegram_defaults(self):
        tg = TelegramConfig()
        assert tg.enabled is False
        assert tg.bot_token == ""
        assert tg.allowed_users == []


class TestConfigLoading:
    """Test loading from actual YAML files."""

    def test_load_minimal_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "general": {"default_target": "en"},
        }))

        config = load_config(str(config_file))
        assert config.general.default_target == "en"
        # Other sections should have defaults
        assert config.stt.engine == "whisper-cpp"
        assert config.translation.model == "gemma3:4b"

    def test_load_full_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        data = {
            "stt": {"engine": "whisper-cpp", "whisper_binary": "/usr/bin/whisper", "whisper_model": "/models/base.bin", "language": "ru"},
            "translation": {"engine": "ollama", "ollama_url": "http://127.0.0.1:11434", "model": "qwen2.5:3b", "timeout": 60},
            "tts": {"engine": "espeak", "piper_voices_dir": "/voices", "espeak_fallback": False},
            "audio": {"sample_rate": 8000, "max_duration": 30},
            "telegram": {"enabled": True, "bot_token": "123:ABC", "allowed_users": [111, 222]},
            "general": {"default_source": "en", "default_target": "ru", "log_level": "DEBUG"},
        }
        config_file.write_text(yaml.dump(data))

        config = load_config(str(config_file))
        assert config.stt.language == "ru"
        assert config.translation.model == "qwen2.5:3b"
        assert config.translation.timeout == 60
        assert config.tts.engine == "espeak"
        assert config.audio.sample_rate == 8000
        assert config.telegram.bot_token == "123:ABC"
        assert config.telegram.allowed_users == [111, 222]
        assert config.general.log_level == "DEBUG"

    def test_unknown_keys_ignored(self, tmp_path):
        """YAML with extra unknown keys should not crash."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "stt": {"engine": "whisper-cpp", "unknown_field": True},
            "future_section": {"foo": "bar"},
        }))

        config = load_config(str(config_file))
        assert config.stt.engine == "whisper-cpp"

    def test_empty_yaml(self, tmp_path):
        """Empty YAML should return defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        config = load_config(str(config_file))
        assert config.general.default_target == "de"

    def test_tilde_expansion(self):
        """Paths with ~ should be expanded."""
        stt = STTConfig(whisper_binary="~/test/whisper-cli")
        assert "~" not in stt.whisper_binary
        assert os.path.expanduser("~") in stt.whisper_binary

    def test_null_voice_preserved(self, tmp_path):
        """null Piper voice (e.g. for Latvian) should become None."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "tts": {"voices": {"lv": None, "de": "de.onnx"}},
        }))

        config = load_config(str(config_file))
        assert config.tts.voices["lv"] is None
        assert config.tts.voices["de"] == "de.onnx"


class TestProjectConfigYaml:
    """Smoke test the actual project config.yaml."""

    def test_project_config_loads(self):
        config = load_config("config.yaml")
        assert config.stt.engine == "whisper-cpp"
        assert config.translation.ollama_url == "http://localhost:11434"
        assert config.tts.piper_voices_dir is not None
        assert config.tts.voices.get("lv") is None
        assert config.general.default_target == "de"
