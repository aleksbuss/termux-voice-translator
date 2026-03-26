"""Telegram bot interface for the offline voice translator.

Uses aiogram 3.x. Handles voice messages (STT -> translate -> TTS)
and text messages (translate -> TTS). Responds with text + voice.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.filters import Command
from aiogram.types import FSInputFile

from src.config import AppConfig
from src.stt import WhisperSTT
from src.translator import LANG_NAMES, OllamaTranslator
from src.tts import EspeakTTS, PiperTTS, TTSManager
from src.audio import AudioManager
from src.pipeline import TranslationPipeline

logger = logging.getLogger(__name__)

router = Router()

# Simple in-memory user state: user_id -> {"target_lang": "de"}
_user_state: dict[int, dict[str, str]] = {}
_pipeline: TranslationPipeline | None = None
_allowed_users: list[int] = []


def _get_target_lang(user_id: int, default: str = "de") -> str:
    return _user_state.get(user_id, {}).get("target_lang", default)


def _set_target_lang(user_id: int, lang: str) -> None:
    if user_id not in _user_state:
        _user_state[user_id] = {}
    _user_state[user_id]["target_lang"] = lang


def _check_access(user_id: int) -> bool:
    if not _allowed_users:
        return True
    return user_id in _allowed_users


@router.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    """Handle /start command."""
    if not _check_access(message.from_user.id):
        return

    langs = ", ".join(f"{code} ({name})" for code, name in sorted(LANG_NAMES.items()))
    await message.answer(
        "Offline Voice Translator\n\n"
        "Send me a voice message or text, and I'll translate it.\n"
        "All processing happens locally on the device — zero internet required.\n\n"
        f"Supported languages: {langs}\n\n"
        "Set target language: /lang de\n"
        "Current target: " + _get_target_lang(message.from_user.id),
    )


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    """Handle /help command."""
    if not _check_access(message.from_user.id):
        return

    await message.answer(
        "Commands:\n"
        "/start — Welcome message\n"
        "/lang <code> — Set target language (e.g. /lang en)\n"
        "/help — This message\n\n"
        "Send a voice message to translate speech.\n"
        "Send text to translate text.\n"
        "Source language is auto-detected."
    )


@router.message(Command("lang"))
async def cmd_lang(message: types.Message) -> None:
    """Handle /lang <code> command."""
    if not _check_access(message.from_user.id):
        return

    parts = message.text.strip().split()
    if len(parts) < 2:
        await message.answer(
            "Usage: /lang <code>\n"
            f"Available: {', '.join(sorted(LANG_NAMES.keys()))}"
        )
        return

    lang = parts[1].lower().strip()
    if lang not in LANG_NAMES:
        await message.answer(
            f"Unknown language: '{lang}'\n"
            f"Available: {', '.join(sorted(LANG_NAMES.keys()))}"
        )
        return

    _set_target_lang(message.from_user.id, lang)
    await message.answer(f"Target language set to: {lang} ({LANG_NAMES[lang]})")


@router.message(F.voice)
async def handle_voice(message: types.Message, bot: Bot) -> None:
    """Handle incoming voice messages."""
    if not _check_access(message.from_user.id):
        return
    if not _pipeline:
        await message.answer("Translator not initialized.")
        return

    tmp_ogg: str | None = None
    tmp_audio: str | None = None

    try:
        target_lang = _get_target_lang(message.from_user.id)
        status = await message.answer("Translating...")

        # Download voice file
        file = await bot.get_file(message.voice.file_id)
        fd, tmp_ogg = tempfile.mkstemp(suffix=".ogg", prefix="tg_voice_")
        os.close(fd)
        await bot.download_file(file.file_path, tmp_ogg)

        # Run pipeline (no playback — we send audio back via Telegram)
        fd2, tmp_audio = tempfile.mkstemp(suffix=".wav", prefix="tg_tts_")
        os.close(fd2)

        result = await _pipeline.translate_voice(
            audio_input=tmp_ogg,
            source_lang="auto",
            target_lang=target_lang,
            play_result=False,
            save_audio=tmp_audio,
        )

        # Send text reply
        text_reply = (
            f"Source ({result.source_language}): {result.source_text}\n"
            f"Translation ({target_lang}): {result.translated_text}"
        )
        await status.edit_text(text_reply)

        # Send voice reply
        if result.audio_path and Path(result.audio_path).is_file():
            voice_file = FSInputFile(result.audio_path)
            await message.answer_voice(voice_file)

    except Exception as exc:
        logger.exception("Error handling voice message")
        await message.answer(f"Error: {exc}")
    finally:
        for tmp in (tmp_ogg, tmp_audio):
            if tmp:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except OSError:
                    pass


@router.message(F.text)
async def handle_text(message: types.Message) -> None:
    """Handle incoming text messages."""
    if not _check_access(message.from_user.id):
        return
    if not _pipeline:
        await message.answer("Translator not initialized.")
        return
    if message.text.startswith("/"):
        return  # Ignore unrecognised commands

    tmp_audio: str | None = None

    try:
        target_lang = _get_target_lang(message.from_user.id)
        status = await message.answer("Translating...")

        fd, tmp_audio = tempfile.mkstemp(suffix=".wav", prefix="tg_tts_")
        os.close(fd)

        result = await _pipeline.translate_text(
            text=message.text,
            source_lang="auto",
            target_lang=target_lang,
            play_result=False,
            save_audio=tmp_audio,
        )

        await status.edit_text(
            f"Translation ({target_lang}): {result.translated_text}"
        )

        if result.audio_path and Path(result.audio_path).is_file():
            voice_file = FSInputFile(result.audio_path)
            await message.answer_voice(voice_file)

    except Exception as exc:
        logger.exception("Error handling text message")
        await message.answer(f"Error: {exc}")
    finally:
        if tmp_audio:
            try:
                Path(tmp_audio).unlink(missing_ok=True)
            except OSError:
                pass


async def run_bot(config: AppConfig) -> None:
    """Start the Telegram bot.

    Args:
        config: Application configuration with Telegram settings.

    Raises:
        ValueError: If bot_token is not configured.
    """
    global _pipeline, _allowed_users

    if not config.telegram.bot_token:
        raise ValueError(
            "Telegram bot token not configured. "
            "Set it in config.yaml under telegram.bot_token"
        )

    _allowed_users = config.telegram.allowed_users

    # Init pipeline components
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

    tts = TTSManager(
        piper=piper,
        espeak=EspeakTTS(),
        espeak_fallback=config.tts.espeak_fallback,
    )
    audio = AudioManager(
        sample_rate=config.audio.sample_rate,
        record_command=config.audio.record_command,
        play_command=config.audio.play_command,
        max_duration=config.audio.max_duration,
    )

    _pipeline = TranslationPipeline(
        stt=stt, translator=translator, tts=tts, audio=audio,
        show_source=False, show_translated=False,
    )

    bot = Bot(token=config.telegram.bot_token)
    dp = Dispatcher()
    dp.include_router(router)

    print("Telegram bot started. Polling...")
    await dp.start_polling(bot)
