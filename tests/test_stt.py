"""Tests for src.stt (WhisperSTT)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.stt import WhisperSTT, STTResult


@pytest.fixture
def stt(tmp_path):
    """Create a WhisperSTT with fake binary and model files."""
    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\necho test")
    binary.chmod(0o755)

    model = tmp_path / "ggml-base.bin"
    model.write_bytes(b"\x00" * 100)

    return WhisperSTT(str(binary), str(model))


def _make_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    """Create a mock async subprocess."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(
        return_value=(stdout.encode(), stderr.encode())
    )
    proc.returncode = returncode
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()
    return proc


@pytest.mark.asyncio
@patch("src.stt.asyncio.create_subprocess_exec")
async def test_transcribe_parses_stdout(mock_exec, stt, tmp_path):
    """Verify text is extracted from whisper stdout."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 2000)

    # ffmpeg proc
    ffmpeg_proc = _make_proc()
    # whisper proc — stdout with text
    whisper_proc = _make_proc(
        stdout="  Hello, how are you?\n",
        stderr="auto-detected language: en\n",
    )
    mock_exec.side_effect = [ffmpeg_proc, whisper_proc]

    result = await stt.transcribe(str(audio), language="auto")

    assert result.text == "Hello, how are you?"
    assert isinstance(result, STTResult)


@pytest.mark.asyncio
@patch("src.stt.asyncio.create_subprocess_exec")
async def test_transcribe_detects_language(mock_exec, stt, tmp_path):
    """Verify language is parsed from whisper stderr."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 2000)

    ffmpeg_proc = _make_proc()
    whisper_proc = _make_proc(
        stdout="Привет\n",
        stderr="auto-detected language: ru\n",
    )
    mock_exec.side_effect = [ffmpeg_proc, whisper_proc]

    result = await stt.transcribe(str(audio), language="auto")

    assert result.language == "ru"


@pytest.mark.asyncio
@patch("src.stt.asyncio.create_subprocess_exec")
async def test_transcribe_converts_audio_first(mock_exec, stt, tmp_path):
    """Verify ffmpeg is called before whisper-cli."""
    audio = tmp_path / "test.ogg"
    audio.write_bytes(b"\x00" * 2000)

    ffmpeg_proc = _make_proc()
    whisper_proc = _make_proc(stdout="text\n", stderr="")
    mock_exec.side_effect = [ffmpeg_proc, whisper_proc]

    await stt.transcribe(str(audio))

    # First call should be ffmpeg, second should be whisper
    calls = mock_exec.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == "ffmpeg"


def test_raises_on_missing_binary(tmp_path):
    """FileNotFoundError when binary doesn't exist."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"\x00")

    with pytest.raises(FileNotFoundError, match="whisper-cli"):
        WhisperSTT("/nonexistent/whisper-cli", str(model))


def test_raises_on_missing_model(tmp_path):
    """FileNotFoundError when model doesn't exist."""
    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh")
    binary.chmod(0o755)

    with pytest.raises(FileNotFoundError, match="Whisper model"):
        WhisperSTT(str(binary), "/nonexistent/model.bin")


@pytest.mark.asyncio
@patch("src.stt.asyncio.create_subprocess_exec")
async def test_transcribe_raises_on_nonzero_exit(mock_exec, stt, tmp_path):
    """RuntimeError on whisper-cli non-zero exit."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"\x00" * 2000)

    ffmpeg_proc = _make_proc()
    whisper_proc = _make_proc(stderr="error", returncode=1)
    mock_exec.side_effect = [ffmpeg_proc, whisper_proc]

    with pytest.raises(RuntimeError, match="exited with code 1"):
        await stt.transcribe(str(audio))
