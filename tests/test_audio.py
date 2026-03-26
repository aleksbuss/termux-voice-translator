"""Tests for src.audio (AudioManager)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.audio import AudioManager


@pytest.fixture
def audio_mgr():
    return AudioManager()


@pytest.mark.asyncio
@patch("src.audio.asyncio.create_subprocess_exec")
async def test_convert_to_wav_16khz_calls_ffmpeg(mock_exec, audio_mgr, tmp_path):
    """Verify ffmpeg is called with correct conversion args."""
    input_file = tmp_path / "input.ogg"
    input_file.write_bytes(b"\x00" * 2000)

    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0
    mock_exec.return_value = proc

    # Mock the output file to exist with valid size
    with patch("src.audio.Path") as mock_path:
        mock_file = mock_path.return_value
        mock_file.is_file.return_value = True
        mock_file.stat.return_value.st_size = 5000
        mock_file.exists.return_value = True

        result = await audio_mgr.convert_to_wav_16khz(str(input_file))

    args = mock_exec.call_args[0]
    assert args[0] == "ffmpeg"
    assert "-ar" in args
    assert "16000" in args
    assert "-ac" in args
    assert "1" in args
    assert "-c:a" in args
    assert "pcm_s16le" in args


@pytest.mark.asyncio
@patch("src.audio.asyncio.create_subprocess_exec")
async def test_convert_raises_on_ffmpeg_failure(mock_exec, audio_mgr, tmp_path):
    """RuntimeError when ffmpeg exits non-zero."""
    input_file = tmp_path / "input.ogg"
    input_file.write_bytes(b"\x00" * 2000)

    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b"error"))
    proc.returncode = 1
    mock_exec.return_value = proc

    with pytest.raises(RuntimeError, match="ffmpeg conversion failed"):
        await audio_mgr.convert_to_wav_16khz(str(input_file))


@pytest.mark.asyncio
@patch("src.audio.shutil.which")
@patch("src.audio.asyncio.create_subprocess_exec")
async def test_play_tries_sox_first(mock_exec, mock_which, audio_mgr, tmp_path):
    """play (sox) is used when available."""
    mock_which.side_effect = lambda cmd: "/usr/bin/play" if cmd == "play" else None

    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0
    mock_exec.return_value = proc

    await audio_mgr.play(str(tmp_path / "test.wav"))

    args = mock_exec.call_args[0]
    assert args[0] == "play"


@pytest.mark.asyncio
@patch("src.audio.shutil.which")
@patch("src.audio.asyncio.create_subprocess_exec")
async def test_play_falls_back_to_termux(mock_exec, mock_which, audio_mgr, tmp_path):
    """Falls back to termux-media-player when sox not available."""
    def which_side_effect(cmd):
        if cmd == "termux-media-player":
            return "/data/data/com.termux/files/usr/bin/termux-media-player"
        return None

    mock_which.side_effect = which_side_effect

    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0
    mock_exec.return_value = proc

    await audio_mgr.play(str(tmp_path / "test.wav"))

    args = mock_exec.call_args[0]
    assert args[0] == "termux-media-player"


def test_get_audio_info(tmp_path):
    """get_audio_info returns correct metadata."""
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"\x00" * 1024)

    mgr = AudioManager()
    info = mgr.get_audio_info(str(wav))

    assert info["size_bytes"] == 1024
    assert info["format"] == "wav"
    assert str(wav) in info["path"]
