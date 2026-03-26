"""CLI entry point for the offline voice translator.

Usage modes:
    python -m src.main                        # interactive (mic -> translate -> speak)
    python -m src.main --to de                # translate to German
    python -m src.main --file recording.ogg   # translate audio file
    python -m src.main --text "Hello"         # translate text directly
    python -m src.main --continuous --to de   # loop mode
    python -m src.main --telegram             # Telegram bot mode
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from src.config import AppConfig, load_config
from src.stt import WhisperSTT
from src.translator import LANG_NAMES, OllamaTranslator
from src.tts import EspeakTTS, PiperTTS, TTSManager
from src.audio import AudioManager
from src.pipeline import TranslationPipeline

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="voice-translator",
        description="Offline voice-to-voice translator. Runs 100% locally on Android/Termux.",
    )
    parser.add_argument(
        "--from", dest="source_lang", default=None,
        help="Source language (ISO code: ru, en, de, ...). Default: auto-detect",
    )
    parser.add_argument(
        "--to", dest="target_lang", default=None,
        help="Target language (ISO code). Default: de (German)",
    )
    parser.add_argument("--file", "-f", help="Input audio file path (skip recording)")
    parser.add_argument("--text", "-t", help="Input text (skip recording + STT)")
    parser.add_argument(
        "--continuous", "-c", action="store_true",
        help="Continuous mode: loop record -> translate -> repeat",
    )
    parser.add_argument("--save", "-s", help="Save translated audio to this path")
    parser.add_argument("--telegram", action="store_true", help="Run as Telegram bot")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio output")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--list-voices", action="store_true", help="List available TTS voices")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    return parser


def init_pipeline(config: AppConfig) -> TranslationPipeline:
    """Initialize all pipeline components from config.

    Args:
        config: Application configuration.

    Returns:
        A fully wired ``TranslationPipeline``.
    """
    stt = WhisperSTT(
        binary_path=config.stt.whisper_binary,
        model_path=config.stt.whisper_model,
    )

    translator = OllamaTranslator(
        base_url=config.translation.ollama_url,
        model=config.translation.model,
        timeout=config.translation.timeout,
    )

    piper: PiperTTS | None = None
    if config.tts.engine == "piper":
        piper = PiperTTS(
            voices_dir=config.tts.piper_voices_dir,
            voice_map=config.tts.voices,
        )

    espeak = EspeakTTS()
    tts = TTSManager(
        piper=piper,
        espeak=espeak,
        espeak_fallback=config.tts.espeak_fallback,
    )

    audio = AudioManager(
        sample_rate=config.audio.sample_rate,
        record_command=config.audio.record_command,
        play_command=config.audio.play_command,
        max_duration=config.audio.max_duration,
    )

    return TranslationPipeline(
        stt=stt,
        translator=translator,
        tts=tts,
        audio=audio,
        show_source=config.general.show_source_text,
        show_translated=config.general.show_translated_text,
    )


async def run_interactive(
    pipeline: TranslationPipeline, source_lang: str, target_lang: str,
    play: bool, save: str | None,
) -> None:
    """Run a single interactive mic -> translate -> speak session."""
    await pipeline.translate_voice(
        audio_input="mic",
        source_lang=source_lang,
        target_lang=target_lang,
        play_result=play,
        save_audio=save,
    )


async def run_file(
    pipeline: TranslationPipeline, file_path: str,
    source_lang: str, target_lang: str,
    play: bool, save: str | None,
) -> None:
    """Translate an audio file."""
    if not Path(file_path).is_file():
        print(f"Error: file not found: {file_path}")
        sys.exit(1)

    await pipeline.translate_voice(
        audio_input=file_path,
        source_lang=source_lang,
        target_lang=target_lang,
        play_result=play,
        save_audio=save,
    )


async def run_text(
    pipeline: TranslationPipeline, text: str,
    source_lang: str, target_lang: str,
    play: bool, save: str | None,
) -> None:
    """Translate plain text (skip STT)."""
    await pipeline.translate_text(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        play_result=play,
        save_audio=save,
    )


async def run_continuous(
    pipeline: TranslationPipeline, source_lang: str, target_lang: str,
    play: bool,
) -> None:
    """Loop: record -> translate -> repeat until Ctrl+C."""
    print("Continuous mode. Press Ctrl+C to stop.\n")
    round_num = 0
    while True:
        round_num += 1
        print(f"--- Round {round_num} ---")
        try:
            await pipeline.translate_voice(
                audio_input="mic",
                source_lang=source_lang,
                target_lang=target_lang,
                play_result=play,
            )
        except Exception as exc:
            logger.error("Round %d failed: %s", round_num, exc)
            print(f"Error: {exc}\nRetrying...\n")
        print()


async def list_voices(config: AppConfig) -> None:
    """Print available TTS voices."""
    print("Piper TTS voices:")
    voices_dir = Path(config.tts.piper_voices_dir)
    for lang, filename in sorted(config.tts.voices.items()):
        if filename:
            model_path = voices_dir / filename
            status = "OK" if model_path.is_file() else "MISSING"
            print(f"  {lang}: {filename} [{status}]")
        else:
            print(f"  {lang}: (no Piper voice, espeak fallback)")

    import shutil
    espeak = "available" if shutil.which("espeak-ng") else "not installed"
    print(f"\nespeak-ng: {espeak}")


async def list_models(config: AppConfig) -> None:
    """Print available Ollama models."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
            resp = await client.get(f"{config.translation.ollama_url}/api/tags")
            resp.raise_for_status()
    except httpx.ConnectError:
        print("Error: Ollama is not running. Start it with: ollama serve &")
        return

    data = resp.json()
    models = data.get("models", [])
    if not models:
        print("No models found. Pull one with: ollama pull gemma3:4b")
        return

    print("Ollama models:")
    for m in models:
        name = m.get("name", "?")
        size = m.get("size", 0)
        size_gb = size / (1024 ** 3) if size else 0
        active = " (configured)" if config.translation.model in name else ""
        print(f"  {name} ({size_gb:.1f} GB){active}")


async def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.general.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # CLI overrides
    source_lang = args.source_lang or config.general.default_source
    target_lang = args.target_lang or config.general.default_target
    play = not args.no_play

    # Info commands (no pipeline needed)
    if args.list_voices:
        await list_voices(config)
        return

    if args.list_models:
        await list_models(config)
        return

    # Telegram bot mode
    if args.telegram:
        from src.telegram_bot import run_bot
        await run_bot(config)
        return

    # Initialize pipeline
    try:
        pipeline = init_pipeline(config)
    except FileNotFoundError as exc:
        print(f"Setup error: {exc}")
        sys.exit(1)

    # Check Ollama health (reuse translator from pipeline)
    health_checker = OllamaTranslator(
        base_url=config.translation.ollama_url,
        model=config.translation.model,
        timeout=config.translation.timeout,
    )
    if not await health_checker.health_check():
        print(
            f"Warning: Ollama model '{config.translation.model}' not available.\n"
            "Make sure Ollama is running (ollama serve &) and the model is pulled "
            f"(ollama pull {config.translation.model}).\n"
        )

    # Dispatch
    if args.text:
        await run_text(pipeline, args.text, source_lang, target_lang, play, args.save)
    elif args.file:
        await run_file(pipeline, args.file, source_lang, target_lang, play, args.save)
    elif args.continuous:
        await run_continuous(pipeline, source_lang, target_lang, play)
    else:
        await run_interactive(pipeline, source_lang, target_lang, play, args.save)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
