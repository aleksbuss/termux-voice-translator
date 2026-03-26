"""Edge case and negative tests for all modules.

Tests error handling, boundary conditions, and unusual inputs
that could occur in production on Termux.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.stt import WhisperSTT, STTResult
from src.translator import OllamaTranslator, TranslationResult, LANG_NAMES
from src.tts import PiperTTS, EspeakTTS, TTSManager, VoiceNotAvailableError
from src.audio import AudioManager
from src.pipeline import TranslationPipeline, PipelineResult


# ── STT edge cases ───────────────────────────────────────────────

class TestSTTEdgeCases:

    def test_nonexistent_audio_file(self, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh")
        binary.chmod(0o755)
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 10)

        stt = WhisperSTT(str(binary), str(model))
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            asyncio.get_event_loop().run_until_complete(
                stt.transcribe("/nonexistent/audio.wav")
            )

    @pytest.mark.asyncio
    @patch("src.stt.asyncio.create_subprocess_exec")
    async def test_whisper_returns_only_whitespace(self, mock_exec, tmp_path):
        """Whisper returning only whitespace/newlines = no speech."""
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh")
        binary.chmod(0o755)
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 10)
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 2000)

        stt = WhisperSTT(str(binary), str(model))

        ffmpeg_proc = AsyncMock()
        ffmpeg_proc.communicate = AsyncMock(return_value=(b"", b""))
        ffmpeg_proc.returncode = 0

        whisper_proc = AsyncMock()
        whisper_proc.communicate = AsyncMock(return_value=(b"  \n  \n", b""))
        whisper_proc.returncode = 0
        whisper_proc.kill = AsyncMock()
        whisper_proc.wait = AsyncMock()

        mock_exec.side_effect = [ffmpeg_proc, whisper_proc]

        result = await stt.transcribe(str(audio))
        assert result.text == ""

    @pytest.mark.asyncio
    @patch("src.stt.asyncio.create_subprocess_exec")
    async def test_whisper_with_timestamp_lines(self, mock_exec, tmp_path):
        """Whisper output with [00:00.000 --> 00:02.000] prefixes."""
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh")
        binary.chmod(0o755)
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 10)
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 2000)

        stt = WhisperSTT(str(binary), str(model))

        ffmpeg_proc = AsyncMock()
        ffmpeg_proc.communicate = AsyncMock(return_value=(b"", b""))
        ffmpeg_proc.returncode = 0

        stdout = b"[00:00.000 --> 00:02.000]  Hello world\n[00:02.000 --> 00:04.000]  How are you\n"
        whisper_proc = AsyncMock()
        whisper_proc.communicate = AsyncMock(return_value=(stdout, b"auto-detected language: en\n"))
        whisper_proc.returncode = 0
        whisper_proc.kill = AsyncMock()
        whisper_proc.wait = AsyncMock()

        mock_exec.side_effect = [ffmpeg_proc, whisper_proc]

        result = await stt.transcribe(str(audio))
        assert "Hello world" in result.text
        assert "How are you" in result.text
        assert "[00:00" not in result.text

    @pytest.mark.asyncio
    @patch("src.stt.asyncio.create_subprocess_exec")
    async def test_ffmpeg_failure(self, mock_exec, tmp_path):
        """FFmpeg failing should raise RuntimeError."""
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh")
        binary.chmod(0o755)
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 10)
        audio = tmp_path / "test.ogg"
        audio.write_bytes(b"\x00" * 2000)

        stt = WhisperSTT(str(binary), str(model))

        ffmpeg_proc = AsyncMock()
        ffmpeg_proc.communicate = AsyncMock(return_value=(b"", b"Invalid data"))
        ffmpeg_proc.returncode = 1

        mock_exec.return_value = ffmpeg_proc

        with pytest.raises(RuntimeError, match="ffmpeg conversion failed"):
            await stt.transcribe(str(audio))


# ── Translator edge cases ────────────────────────────────────────

class TestTranslatorEdgeCases:

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_translate_single_word(self, mock_client_cls):
        """Single word input should produce single word output."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Hund"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator()
        result = await tr.translate("dog", "en", "de")
        assert result.text == "Hund"

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_translate_unknown_language_code(self, mock_client_cls):
        """Unknown language code used as-is in prompt (not crashing)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "translated"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator()
        result = await tr.translate("hello", "xx", "yy")
        # Should use raw codes since they're not in LANG_NAMES
        assert result.source_lang == "xx"

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_translate_timeout(self, mock_client_cls):
        """TimeoutError with actionable message."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator(timeout=5)
        with pytest.raises(TimeoutError, match="timed out"):
            await tr.translate("hello", "en", "de")

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_health_check_connection_refused(self, mock_client_cls):
        """health_check returns False on connection error."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator()
        assert await tr.health_check() is False

    @pytest.mark.asyncio
    @patch("src.translator.httpx.AsyncClient")
    async def test_health_check_model_with_tag(self, mock_client_cls):
        """Model name with :latest tag should still match."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "gemma3:4b"}]
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        tr = OllamaTranslator(model="gemma3:4b")
        assert await tr.health_check() is True


# ── TTS edge cases ───────────────────────────────────────────────

class TestTTSEdgeCases:

    def test_piper_model_file_missing_on_disk(self, tmp_path):
        """VoiceNotAvailableError when .onnx file doesn't exist."""
        piper = PiperTTS(str(tmp_path), {"de": "de_DE-thorsten-high.onnx"})
        with pytest.raises(VoiceNotAvailableError, match="not found"):
            piper._get_voice("de")

    def test_piper_empty_voice_map(self, tmp_path):
        piper = PiperTTS(str(tmp_path), {})
        with pytest.raises(VoiceNotAvailableError):
            piper._get_voice("de")

    @pytest.mark.asyncio
    async def test_tts_manager_no_fallback_raises(self):
        """Without espeak_fallback, missing Piper voice raises."""
        mock_piper = AsyncMock(spec=PiperTTS)
        mock_piper.synthesize = AsyncMock(
            side_effect=VoiceNotAvailableError("no voice")
        )

        mgr = TTSManager(piper=mock_piper, espeak=EspeakTTS(), espeak_fallback=False)
        with pytest.raises(VoiceNotAvailableError):
            await mgr.synthesize("test", "xx", "/tmp/out.wav")

    @pytest.mark.asyncio
    async def test_tts_manager_no_piper_no_fallback(self):
        """No Piper + no fallback should raise."""
        mgr = TTSManager(piper=None, espeak=EspeakTTS(), espeak_fallback=False)
        with pytest.raises(VoiceNotAvailableError, match="not configured"):
            await mgr.synthesize("test", "de", "/tmp/out.wav")

    @pytest.mark.asyncio
    @patch("src.tts.asyncio.create_subprocess_exec")
    async def test_espeak_nonzero_exit(self, mock_exec):
        """espeak-ng failing raises RuntimeError."""
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b"voice not found"))
        proc.returncode = 1
        proc.kill = AsyncMock()
        proc.wait = AsyncMock()
        mock_exec.return_value = proc

        espeak = EspeakTTS()
        with pytest.raises(RuntimeError, match="exited with code 1"):
            await espeak.synthesize("test", "xx", "/tmp/out.wav")


# ── Audio edge cases ─────────────────────────────────────────────

class TestAudioEdgeCases:

    @pytest.mark.asyncio
    @patch("src.audio.shutil.which", return_value=None)
    async def test_record_no_tool_available(self, mock_which):
        """RuntimeError when no recording tool is found."""
        mgr = AudioManager()
        with pytest.raises(RuntimeError, match="No recording tool found"):
            await mgr.record("/tmp/out.wav")

    @pytest.mark.asyncio
    @patch("src.audio.shutil.which", return_value=None)
    async def test_play_no_player_available(self, mock_which):
        """No crash when no player is available — just logs warning."""
        mgr = AudioManager()
        # Should not raise, just log
        await mgr.play("/tmp/test.wav")

    def test_get_audio_info_nonexistent_file(self):
        """get_audio_info handles missing files gracefully."""
        mgr = AudioManager()
        info = mgr.get_audio_info("/nonexistent/file.wav")
        assert info["size_bytes"] == 0

    def test_get_audio_info_format_detection(self, tmp_path):
        """Format is extracted from file extension."""
        for ext in ("wav", "ogg", "mp3", "m4a", "flac"):
            f = tmp_path / f"test.{ext}"
            f.write_bytes(b"\x00")
            mgr = AudioManager()
            assert mgr.get_audio_info(str(f))["format"] == ext


# ── Pipeline edge cases ──────────────────────────────────────────

class TestPipelineEdgeCases:

    @pytest.mark.asyncio
    async def test_translate_text_empty_string(self):
        """Empty input text still runs through translation."""
        stt = AsyncMock()
        translator = AsyncMock()
        translator.translate = AsyncMock(
            return_value=TranslationResult(text="", source_lang="en", target_lang="de")
        )
        tts = AsyncMock()
        tts.synthesize = AsyncMock(return_value="/tmp/out.wav")
        audio = AsyncMock()

        pipeline = TranslationPipeline(
            stt=stt, translator=translator, tts=tts, audio=audio,
            show_source=False, show_translated=False,
        )

        result = await pipeline.translate_text("", "en", "de", play_result=False)
        assert result.translated_text == ""
        stt.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_stt_error_propagates(self, tmp_path):
        """STT error should propagate, not be swallowed."""
        stt = AsyncMock()
        stt.transcribe = AsyncMock(side_effect=RuntimeError("whisper crashed"))
        translator = AsyncMock()
        tts = AsyncMock()
        audio = AsyncMock()

        pipeline = TranslationPipeline(
            stt=stt, translator=translator, tts=tts, audio=audio,
            show_source=False, show_translated=False,
        )

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        with pytest.raises(RuntimeError, match="whisper crashed"):
            await pipeline.translate_voice(str(audio_file), "auto", "de", play_result=False)

    @pytest.mark.asyncio
    async def test_pipeline_translation_error_propagates(self, tmp_path):
        """Translation error should propagate after successful STT."""
        stt = AsyncMock()
        stt.transcribe = AsyncMock(
            return_value=STTResult(text="hello", language="en")
        )
        translator = AsyncMock()
        translator.translate = AsyncMock(
            side_effect=ConnectionError("Ollama not running")
        )
        tts = AsyncMock()
        audio = AsyncMock()

        pipeline = TranslationPipeline(
            stt=stt, translator=translator, tts=tts, audio=audio,
            show_source=False, show_translated=False,
        )

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        with pytest.raises(ConnectionError, match="Ollama"):
            await pipeline.translate_voice(str(audio_file), "auto", "de", play_result=False)


# ── Text splitting for long TTS input ────────────────────────────

class TestTextSplitting:

    def test_split_sentences_basic(self):
        from src.tts import _split_sentences
        text = "Hello world. How are you? I am fine! Thanks."
        chunks = _split_sentences(text)
        assert len(chunks) >= 1
        # All original text preserved
        combined = " ".join(chunks)
        assert "Hello world." in combined
        assert "Thanks." in combined

    def test_split_sentences_single_sentence(self):
        from src.tts import _split_sentences
        text = "Short text"
        chunks = _split_sentences(text)
        assert chunks == ["Short text"]

    def test_split_sentences_very_long(self):
        from src.tts import _split_sentences
        # Create text >500 chars with sentences
        text = ". ".join([f"Sentence number {i}" for i in range(50)]) + "."
        assert len(text) > 500
        chunks = _split_sentences(text)
        assert len(chunks) > 1
        # Each chunk should be <= 500 chars
        for chunk in chunks:
            assert len(chunk) <= 500
