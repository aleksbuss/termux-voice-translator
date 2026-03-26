"""Configuration loader for Termux offline voice translator.

Loads config.yaml with sane defaults using dataclasses.
If the config file is missing, all defaults are applied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _expand(path: str) -> str:
    """Expand ~ and environment variables in a path string."""
    return os.path.expandvars(os.path.expanduser(path))


@dataclass
class STTConfig:
    """Speech-to-text configuration."""

    engine: str = "whisper-cpp"
    whisper_binary: str = "~/whisper.cpp/build/bin/whisper-cli"
    whisper_model: str = "~/whisper.cpp/models/ggml-base.bin"
    language: str = "auto"

    def __post_init__(self) -> None:
        self.whisper_binary = _expand(self.whisper_binary)
        self.whisper_model = _expand(self.whisper_model)


@dataclass
class TranslationConfig:
    """Translation engine configuration."""

    engine: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    model: str = "gemma3:4b"
    timeout: int = 120


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""

    engine: str = "piper"
    piper_voices_dir: str = "~/voice-translator/models/piper"
    espeak_fallback: bool = True
    voices: dict[str, str] = field(default_factory=lambda: {
        "de": "de_DE-thorsten-high.onnx",
        "en": "en_US-lessac-medium.onnx",
        "ru": "ru_RU-denis-medium.onnx",
        "es": "es_ES-sharvard-medium.onnx",
        "fr": "fr_FR-siwis-medium.onnx",
    })
    speed: float = 1.0

    def __post_init__(self) -> None:
        self.piper_voices_dir = _expand(self.piper_voices_dir)


@dataclass
class AudioConfig:
    """Audio recording and playback configuration."""

    sample_rate: int = 16000
    format: str = "wav"
    record_command: str = "termux-microphone-record"
    play_command: str = "play"
    max_duration: int = 60


@dataclass
class TelegramConfig:
    """Telegram bot integration configuration."""

    enabled: bool = False
    bot_token: str = ""
    allowed_users: list[int] = field(default_factory=list)


@dataclass
class GeneralConfig:
    """General application settings."""

    default_source: str = "auto"
    default_target: str = "de"
    show_source_text: bool = True
    show_translated_text: bool = True
    log_level: str = "INFO"


@dataclass
class AppConfig:
    """Top-level application configuration.

    Aggregates all configuration sections. Constructed via
    ``load_config()`` which merges a YAML file (if present)
    with built-in defaults.
    """

    stt: STTConfig = field(default_factory=STTConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)


# Mapping from YAML section names to their dataclass types.
_SECTION_MAP: dict[str, type] = {
    "stt": STTConfig,
    "translation": TranslationConfig,
    "tts": TTSConfig,
    "audio": AudioConfig,
    "telegram": TelegramConfig,
    "general": GeneralConfig,
}


def _build_section(cls: type, raw: dict[str, Any] | None) -> Any:
    """Instantiate a config dataclass from a raw dict, ignoring unknown keys."""
    if not raw:
        return cls()
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in known}
    return cls(**filtered)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load application configuration from a YAML file.

    Reads the given YAML file and merges its values with built-in
    defaults.  If the file does not exist, a fully-defaulted
    ``AppConfig`` is returned.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        A populated ``AppConfig`` instance.
    """
    config_path = Path(_expand(path))

    if not config_path.is_file():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    sections: dict[str, Any] = {}
    for section_name, cls in _SECTION_MAP.items():
        sections[section_name] = _build_section(cls, raw.get(section_name))

    return AppConfig(**sections)
