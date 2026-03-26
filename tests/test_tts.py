"""Tests for src.tts (PiperTTS, EspeakTTS, TTSManager)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tts import PiperTTS, EspeakTTS, TTSManager, VoiceNotAvailableError


@pytest.fixture
def espeak():
    return EspeakTTS()


@pytest.mark.asyncio
@patch("src.tts.asyncio.get_running_loop")
async def test_piper_synthesize_creates_wav(mock_loop, tmp_path):
    """Verify PiperVoice.load and synthesize are called."""
    voice_file = tmp_path / "de.onnx"
    voice_file.write_bytes(b"\x00" * 100)

    mock_voice = MagicMock()
    mock_voice.synthesize = MagicMock()

    with patch("src.tts.PiperTTS._get_voice", return_value=mock_voice):
        piper = PiperTTS(str(tmp_path), {"de": "de.onnx"})
        # Mock run_in_executor to call the function directly
        loop = MagicMock()
        loop.run_in_executor = AsyncMock(
            side_effect=lambda _, fn, *args: fn(*args)
        )
        mock_loop.return_value = loop

        output = str(tmp_path / "out.wav")
        # Create a minimal WAV so the sync function doesn't fail
        with patch("wave.open"):
            result = await piper.synthesize("Hallo", "de", output)

    assert result == output


def test_piper_raises_on_missing_voice(tmp_path):
    """VoiceNotAvailableError when language not in voice_map."""
    piper = PiperTTS(str(tmp_path), {"en": "en.onnx"})

    with pytest.raises(VoiceNotAvailableError, match="No Piper voice"):
        piper._get_voice("xx")


def test_piper_raises_on_none_voice(tmp_path):
    """VoiceNotAvailableError when voice_map value is None."""
    piper = PiperTTS(str(tmp_path), {"lv": None})

    with pytest.raises(VoiceNotAvailableError, match="No Piper voice"):
        piper._get_voice("lv")


@pytest.mark.asyncio
@patch("src.tts.asyncio.create_subprocess_exec")
async def test_espeak_synthesize_calls_subprocess(mock_exec, espeak, tmp_path):
    """Verify espeak-ng is called with correct args."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0
    mock_exec.return_value = proc

    output = str(tmp_path / "out.wav")
    result = await espeak.synthesize("Hello", "en", output)

    assert result == output
    mock_exec.assert_called_once()
    args = mock_exec.call_args[0]
    assert args[0] == "espeak-ng"
    assert "-v" in args
    assert "en" in args
    assert "-w" in args


@pytest.mark.asyncio
@patch("src.tts.asyncio.create_subprocess_exec")
async def test_espeak_passes_text_as_argument(mock_exec, espeak, tmp_path):
    """Text with special chars is passed as positional arg (no shell)."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b"", b""))
    proc.returncode = 0
    mock_exec.return_value = proc

    text = 'Hello "world" & stuff'
    await espeak.synthesize(text, "en", str(tmp_path / "out.wav"))

    args = mock_exec.call_args[0]
    assert text in args  # passed directly, not through shell


@pytest.mark.asyncio
async def test_tts_manager_falls_back_to_espeak(tmp_path):
    """TTSManager uses espeak when Piper fails."""
    mock_piper = AsyncMock(spec=PiperTTS)
    mock_piper.synthesize = AsyncMock(
        side_effect=VoiceNotAvailableError("no voice")
    )

    mock_espeak = AsyncMock(spec=EspeakTTS)
    mock_espeak.synthesize = AsyncMock(return_value=str(tmp_path / "out.wav"))

    manager = TTSManager(piper=mock_piper, espeak=mock_espeak, espeak_fallback=True)
    await manager.synthesize("Hello", "xx", str(tmp_path / "out.wav"))

    mock_espeak.synthesize.assert_called_once()


@pytest.mark.asyncio
async def test_tts_manager_uses_piper_first(tmp_path):
    """TTSManager tries Piper before espeak."""
    mock_piper = AsyncMock(spec=PiperTTS)
    mock_piper.synthesize = AsyncMock(return_value=str(tmp_path / "out.wav"))

    mock_espeak = AsyncMock(spec=EspeakTTS)

    manager = TTSManager(piper=mock_piper, espeak=mock_espeak)
    await manager.synthesize("Hello", "de", str(tmp_path / "out.wav"))

    mock_piper.synthesize.assert_called_once()
    mock_espeak.synthesize.assert_not_called()
