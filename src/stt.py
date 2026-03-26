"""Speech-to-Text module wrapping whisper.cpp CLI binary.

Calls the locally compiled ``whisper-cli`` via subprocess.
Does NOT use Python whisper or faster-whisper.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

STT_TIMEOUT_SEC = 60


@dataclass
class STTResult:
    """Result of a speech-to-text transcription.

    Attributes:
        text: Transcribed text.
        language: Detected or specified ISO 639-1 language code.
        duration_sec: Duration of the audio in seconds.
    """

    text: str
    language: str
    duration_sec: float = 0.0


class WhisperSTT:
    """Speech-to-Text using locally compiled whisper.cpp.

    whisper-cli expects WAV input: 16 kHz, mono, 16-bit PCM.
    Any other format is converted via ffmpeg before passing to whisper.

    Args:
        binary_path: Path to the ``whisper-cli`` executable.
        model_path: Path to the GGML model file (e.g. ``ggml-base.bin``).

    Raises:
        FileNotFoundError: If the binary or model file does not exist.
    """

    def __init__(self, binary_path: str, model_path: str) -> None:
        self._binary = Path(os.path.expanduser(binary_path))
        self._model = Path(os.path.expanduser(model_path))

        if not self._binary.is_file():
            raise FileNotFoundError(
                f"whisper-cli binary not found: {self._binary}\n"
                "Compile it from source: cd ~/whisper.cpp && mkdir -p build && "
                "cd build && cmake .. && make -j$(nproc)"
            )
        if not self._model.is_file():
            raise FileNotFoundError(
                f"Whisper model not found: {self._model}\n"
                f"Download it: wget -O {self._model} "
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
            )

    async def transcribe(
        self, audio_path: str, language: str = "auto"
    ) -> STTResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (any format — converted to WAV).
            language: ISO 639-1 code or ``"auto"`` for detection.

        Returns:
            An ``STTResult`` with transcribed text and detected language.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If whisper-cli returns a non-zero exit code.
            TimeoutError: If transcription exceeds 60 seconds.
        """
        audio = Path(audio_path)
        if not audio.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        tmp_wav: str | None = None
        try:
            # Step 1: convert to 16 kHz mono WAV
            tmp_wav = await self._convert_to_wav(audio_path)

            # Step 2: get WAV duration for timing info
            duration_sec = self._get_wav_duration(tmp_wav)

            # Step 3: run whisper-cli
            text, detected_lang = await self._run_whisper(tmp_wav, language)

            return STTResult(
                text=text.strip(),
                language=detected_lang if language == "auto" else language,
                duration_sec=duration_sec,
            )
        finally:
            if tmp_wav and Path(tmp_wav).exists():
                try:
                    Path(tmp_wav).unlink()
                except OSError:
                    pass

    @staticmethod
    def _get_wav_duration(wav_path: str) -> float:
        """Get duration of a WAV file in seconds."""
        import wave
        try:
            with wave.open(wav_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate if rate > 0 else 0.0
        except Exception:
            return 0.0

    async def _convert_to_wav(self, input_path: str) -> str:
        """Convert any audio file to whisper-compatible WAV."""
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="whisper_")
        os.close(fd)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            tmp_path, "-y",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg conversion failed (exit {proc.returncode}): {err}")

        logger.debug("Converted %s -> %s", input_path, tmp_path)
        return tmp_path

    async def _run_whisper(
        self, wav_path: str, language: str
    ) -> tuple[str, str]:
        """Run whisper-cli and parse output."""
        cmd = [
            str(self._binary),
            "--model", str(self._model),
            "--language", language,
            "--no-timestamps",
            "--file", wav_path,
        ]

        logger.debug("Running: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=STT_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"whisper-cli timed out after {STT_TIMEOUT_SEC}s"
            ) from None

        if proc.returncode != 0:
            err = stderr_bytes.decode(errors="replace").strip()
            raise RuntimeError(
                f"whisper-cli exited with code {proc.returncode}: {err}"
            )

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")

        # Parse transcribed text (strip timing prefixes like [00:00:00.000 --> ...])
        lines = []
        for line in stdout.strip().splitlines():
            cleaned = re.sub(r"^\[.*?\]\s*", "", line).strip()
            if cleaned:
                lines.append(cleaned)
        text = " ".join(lines)

        # Parse detected language from stderr
        detected = "unknown"
        lang_match = re.search(r"auto-detected language:\s*(\w+)", stderr)
        if lang_match:
            detected = lang_match.group(1)

        logger.info("Whisper: lang=%s, text_len=%d", detected, len(text))
        return text, detected
