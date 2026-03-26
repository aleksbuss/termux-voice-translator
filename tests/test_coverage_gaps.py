"""Tests covering gaps identified by the test audit.

Fills missing coverage: timeout tests, WAV concatenation,
recording tools, TTS voice loading errors, and install.sh validation.
"""

from __future__ import annotations

import asyncio
import struct
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tts import EspeakTTS, PiperTTS, _concatenate_wavs, _split_sentences
from src.translator import OllamaTranslator
from src.audio import AudioManager
from src.stt import WhisperSTT


# ── Translator timeout tests ─────────────────────────────────────

class TestTranslatorTimeout:

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_translate_read_timeout(self, mock_client_cls):
        """ReadTimeout raises TimeoutError with actionable message."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator(timeout=5)
        with pytest.raises(TimeoutError, match="timed out"):
            await tr.translate("hello", "en", "de")

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_translate_pool_timeout(self, mock_client_cls):
        """PoolTimeout raises TimeoutError."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.PoolTimeout("pool timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator()
        with pytest.raises(TimeoutError):
            await tr.translate("hello", "en", "de")


# ── Translator HTTP status error tests ────────────────────────────

class TestTranslatorHTTPErrors:

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_translate_http_404(self, mock_client_cls):
        """HTTP 404 from Ollama raises RuntimeError."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        error = httpx.HTTPStatusError("not found", request=MagicMock(), response=mock_resp)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=error)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator()
        with pytest.raises(RuntimeError, match="HTTP 404"):
            await tr.translate("hello", "en", "de")

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_translate_invalid_json_response(self, mock_client_cls):
        """Invalid JSON from Ollama raises ValueError."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("invalid json")
        mock_resp.text = "not json at all"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator()
        with pytest.raises(ValueError, match="invalid JSON"):
            await tr.translate("hello", "en", "de")


# ── EspeakTTS timeout test ───────────────────────────────────────

class TestEspeakTimeout:

    @pytest.mark.asyncio
    @patch("src.tts.asyncio.create_subprocess_exec")
    async def test_espeak_timeout_kills_process(self, mock_exec):
        """espeak-ng timeout kills subprocess and raises TimeoutError."""
        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        proc.kill = AsyncMock()
        proc.wait = AsyncMock()
        mock_exec.return_value = proc

        espeak = EspeakTTS()
        with pytest.raises(TimeoutError, match="espeak-ng timed out"):
            await espeak.synthesize("test", "en", "/tmp/out.wav")

        proc.kill.assert_called_once()


# ── WAV concatenation tests ──────────────────────────────────────

def _create_wav(path: str, num_frames: int = 100, sample_rate: int = 22050) -> None:
    """Create a minimal valid WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence frames
        wf.writeframes(b"\x00\x00" * num_frames)


class TestWavConcatenation:

    def test_concatenate_two_wavs(self, tmp_path):
        part1 = str(tmp_path / "p1.wav")
        part2 = str(tmp_path / "p2.wav")
        output = str(tmp_path / "out.wav")

        _create_wav(part1, num_frames=50)
        _create_wav(part2, num_frames=75)

        _concatenate_wavs([part1, part2], output)

        with wave.open(output, "rb") as wf:
            assert wf.getnframes() == 125  # 50 + 75

    def test_concatenate_single_wav(self, tmp_path):
        part = str(tmp_path / "p.wav")
        output = str(tmp_path / "out.wav")
        _create_wav(part, num_frames=100)

        _concatenate_wavs([part], output)

        with wave.open(output, "rb") as wf:
            assert wf.getnframes() == 100

    def test_concatenate_empty_list_raises(self):
        with pytest.raises(ValueError, match="No WAV parts"):
            _concatenate_wavs([], "/tmp/out.wav")

    def test_concatenate_preserves_sample_rate(self, tmp_path):
        part1 = str(tmp_path / "p1.wav")
        part2 = str(tmp_path / "p2.wav")
        output = str(tmp_path / "out.wav")

        _create_wav(part1, sample_rate=16000)
        _create_wav(part2, sample_rate=16000)

        _concatenate_wavs([part1, part2], output)

        with wave.open(output, "rb") as wf:
            assert wf.getframerate() == 16000


# ── Audio recording tool detection ────────────────────────────────

class TestRecordingToolDetection:

    @pytest.mark.asyncio
    @patch("src.audio.shutil.which")
    @patch("src.audio.asyncio.create_subprocess_exec")
    async def test_record_uses_termux_when_available(self, mock_exec, mock_which, tmp_path):
        """termux-microphone-record is preferred when available."""
        mock_which.side_effect = lambda cmd: (
            "/usr/bin/termux-microphone-record" if cmd == "termux-microphone-record" else None
        )

        # Mock the recording process
        rec_proc = AsyncMock()
        rec_proc.communicate = AsyncMock(return_value=(b"", b""))
        rec_proc.returncode = 0
        rec_proc.wait = AsyncMock()

        # Mock the stop process
        stop_proc = AsyncMock()
        stop_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_exec.side_effect = [rec_proc, stop_proc]

        mgr = AudioManager()
        output = str(tmp_path / "rec.wav")

        # Mock stdin.readline to simulate Enter press
        with patch("src.audio.sys.stdin") as mock_stdin:
            mock_stdin.readline = MagicMock(return_value="\n")
            await mgr.record(output, max_duration=5)

        # First call should be termux-microphone-record
        first_call_args = mock_exec.call_args_list[0][0]
        assert first_call_args[0] == "termux-microphone-record"

    @pytest.mark.asyncio
    @patch("src.audio.shutil.which")
    @patch("src.audio.asyncio.create_subprocess_exec")
    async def test_record_uses_arecord_fallback(self, mock_exec, mock_which, tmp_path):
        """arecord is used when termux-microphone-record not available."""
        mock_which.side_effect = lambda cmd: (
            "/usr/bin/arecord" if cmd == "arecord" else None
        )

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        mgr = AudioManager()
        output = str(tmp_path / "rec.wav")
        await mgr.record(output, max_duration=5)

        first_call_args = mock_exec.call_args[0]
        assert first_call_args[0] == "arecord"


# ── STT duration calculation test ─────────────────────────────────

class TestSTTDuration:

    def test_get_wav_duration(self, tmp_path):
        """Verify WAV duration calculation."""
        wav_path = str(tmp_path / "test.wav")
        _create_wav(wav_path, num_frames=16000, sample_rate=16000)

        # Use internal method
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh")
        binary.chmod(0o755)
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 10)

        stt = WhisperSTT(str(binary), str(model))
        duration = stt._get_wav_duration(wav_path)
        assert abs(duration - 1.0) < 0.01  # 16000 frames / 16000 Hz = 1.0 sec

    def test_get_wav_duration_invalid_file(self, tmp_path):
        """Invalid file returns 0.0 instead of crashing."""
        bad_file = str(tmp_path / "bad.wav")
        with open(bad_file, "wb") as f:
            f.write(b"not a wav file")

        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh")
        binary.chmod(0o755)
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 10)

        stt = WhisperSTT(str(binary), str(model))
        assert stt._get_wav_duration(bad_file) == 0.0


# ── Text splitting edge cases ────────────────────────────────────

class TestSplitSentencesEdgeCases:

    def test_empty_string(self):
        result = _split_sentences("")
        assert result == [""]

    def test_no_punctuation(self):
        """Text without sentence-ending punctuation stays as one chunk."""
        result = _split_sentences("hello world this is a test")
        assert result == ["hello world this is a test"]

    def test_unicode_text(self):
        """Russian text with punctuation splits correctly."""
        text = "Привет мир. Как дела? Хорошо!"
        result = _split_sentences(text)
        combined = " ".join(result)
        assert "Привет мир." in combined
        assert "Хорошо!" in combined

    def test_single_punctuation(self):
        result = _split_sentences("Hello.")
        assert result == ["Hello."]


# ── Install.sh validation ─────────────────────────────────────────

class TestInstallScript:

    def test_install_sh_is_executable(self):
        """install.sh has correct permissions."""
        import os
        import stat
        st = os.stat("install.sh")
        assert st.st_mode & stat.S_IXUSR  # Owner execute bit

    def test_install_sh_has_correct_shebang(self):
        """install.sh uses Termux bash shebang."""
        with open("install.sh") as f:
            first_line = f.readline().strip()
        assert first_line == "#!/data/data/com.termux/files/usr/bin/bash"

    def test_install_sh_has_set_euo_pipefail(self):
        """install.sh uses strict error handling."""
        with open("install.sh") as f:
            content = f.read()
        assert "set -euo pipefail" in content

    def test_install_sh_uses_command_v_not_which(self):
        """install.sh uses 'command -v' not 'which' (which doesn't exist in Termux)."""
        with open("install.sh") as f:
            content = f.read()
        assert "which " not in content  # no bare 'which' command
        assert "command -v" in content
