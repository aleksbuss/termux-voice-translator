"""Audio utilities for recording, format conversion, and playback.

All operations use subprocess calls to ffmpeg and Termux audio tools.
Tool availability is checked via ``shutil.which()`` (not the ``which``
command, which does not exist in Termux).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioManager:
    """Handles audio recording, format conversion, and playback.

    Args:
        sample_rate: Target sample rate for conversions.
        record_command: Preferred recording tool name.
        play_command: Preferred playback tool name.
        max_duration: Maximum recording duration in seconds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        record_command: str = "termux-microphone-record",
        play_command: str = "play",
        max_duration: int = 60,
    ) -> None:
        self._sample_rate = sample_rate
        self._record_command = record_command
        self._play_command = play_command
        self._max_duration = max_duration

    async def record(
        self, output_path: str, max_duration: int | None = None
    ) -> str:
        """Record audio from the microphone.

        Tries recording tools in order of preference:
        1. ``termux-microphone-record`` (Termux:API)
        2. ``arecord`` (pulseaudio)

        Args:
            output_path: Where to save the recording.
            max_duration: Override max recording duration.

        Returns:
            Path to the recorded audio file.

        Raises:
            RuntimeError: If no recording tool is available.
        """
        duration = max_duration or self._max_duration

        if shutil.which("termux-microphone-record"):
            return await self._record_termux(output_path, duration)
        elif shutil.which("arecord"):
            return await self._record_arecord(output_path, duration)
        else:
            raise RuntimeError(
                "No recording tool found. Install one of:\n"
                "  - Termux:API from F-Droid (provides termux-microphone-record)\n"
                "  - pkg install pulseaudio (provides arecord)"
            )

    async def _record_termux(self, output_path: str, max_duration: int) -> str:
        """Record using termux-microphone-record."""
        print(f"Recording... Press Enter to stop (max {max_duration}s)")

        proc = await asyncio.create_subprocess_exec(
            "termux-microphone-record",
            "-f", output_path,
            "-l", str(max_duration),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for Enter or timeout
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, sys.stdin.readline),
                timeout=max_duration,
            )
        except asyncio.TimeoutError:
            pass

        # Stop recording
        stop = await asyncio.create_subprocess_exec(
            "termux-microphone-record", "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await stop.communicate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning("termux-microphone-record did not exit after stop, killed")

        logger.info("Recorded to %s via termux-microphone-record", output_path)
        return output_path

    async def _record_arecord(self, output_path: str, max_duration: int) -> str:
        """Record using arecord."""
        print(f"Recording... (max {max_duration}s, Ctrl+C to stop)")

        proc = await asyncio.create_subprocess_exec(
            "arecord",
            "-f", "S16_LE",
            "-r", str(self._sample_rate),
            "-c", "1",
            "-d", str(max_duration),
            output_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"arecord failed with exit code {proc.returncode}")

        logger.info("Recorded to %s via arecord", output_path)
        return output_path

    async def convert_to_wav_16khz(self, input_path: str) -> str:
        """Convert any audio file to whisper-compatible WAV.

        Output: 16 kHz, mono, 16-bit PCM, WAV format.

        Args:
            input_path: Path to the source audio file.

        Returns:
            Path to the converted temporary WAV file.

        Raises:
            RuntimeError: If ffmpeg conversion fails or output is invalid.
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="audio_conv_")
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

        output = Path(tmp_path)
        if not output.is_file() or output.stat().st_size < 1000:
            raise RuntimeError(
                f"ffmpeg produced invalid output: {tmp_path} "
                f"(size={output.stat().st_size if output.exists() else 0})"
            )

        logger.debug("Converted %s -> %s", input_path, tmp_path)
        return tmp_path

    async def play(self, audio_path: str) -> None:
        """Play an audio file through the device speakers.

        Tries playback tools in order of preference:
        1. ``play`` (sox)
        2. ``termux-media-player``
        3. ``ffplay``

        Args:
            audio_path: Path to the audio file to play.
        """
        if shutil.which("play"):
            cmd = ["play", audio_path]
        elif shutil.which("termux-media-player"):
            cmd = ["termux-media-player", "play", audio_path]
        elif shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", audio_path]
        else:
            logger.warning(
                "No audio player found. Install one of: sox, termux-api, ffmpeg"
            )
            return

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning("Audio playback timed out after 120s")

    def get_audio_info(self, file_path: str) -> dict[str, str | int]:
        """Get basic information about an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Dict with ``path``, ``size_bytes``, and ``format`` keys.
        """
        p = Path(file_path)
        return {
            "path": str(p),
            "size_bytes": p.stat().st_size if p.is_file() else 0,
            "format": p.suffix.lstrip(".") if p.suffix else "unknown",
        }
