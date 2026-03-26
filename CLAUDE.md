# CLAUDE.md — Termux Offline Voice Translator

## Project identity

**Name:** `termux-voice-translator`
**One-liner:** Fully offline voice-to-voice translator running on Android via Termux. Speak in one language → hear the translation spoken back. Zero internet. Zero API costs. Zero data leaks.
**Author:** Aleksejs Buss (`github.com/aleksbuss`)
**License:** MIT
**Target platform:** Android (ARM64/aarch64) running Termux from F-Droid

---

## CRITICAL: Termux-specific constraints

You are NOT building for standard Linux. Termux on Android has hard constraints that WILL break your code if ignored. Read every line below.

### The environment

- **OS:** Android (not Linux). Termux provides a Linux-like userspace via `$PREFIX` (`/data/data/com.termux/files/usr`).
- **Architecture:** ARM64 (aarch64). All native binaries must be ARM64.
- **libc:** Android Bionic, NOT glibc. Pre-compiled Linux binaries that expect `/lib/ld-linux-aarch64.so.1` WILL NOT RUN. This means:
  - ❌ Piper TTS pre-built binary from GitHub releases — DOES NOT WORK (ELF interpreter mismatch)
  - ✅ Piper TTS via `pip install piper-tts` — WORKS (Python wrapper, no native binary dependency)
  - ✅ whisper.cpp compiled from source inside Termux — WORKS (cmake + make, compiles against Bionic)
  - ✅ Ollama installed via `curl -fsSL https://ollama.com/install.sh | sh` — WORKS on Termux
- **No `which` command.** Use `shutil.which()` in Python or `command -v` in Bash instead.
- **No `sudo`.** Termux is not rooted. All paths are under `$HOME` or `$PREFIX`.
- **No systemd.** Use `nohup`, `&`, or `.bashrc` autostart for background processes.
- **Storage:** User must run `termux-setup-storage` first for access to `/sdcard`.
- **Python:** Installed via `pkg install python`. pip packages that need compilation may need `pkg install build-essential` first.
- **FFmpeg:** Available via `pkg install ffmpeg`. Required for audio format conversion.

### Proven working components (from prior projects)

1. **whisper.cpp** — compiled from source via cmake in Termux. Binary path after build: `~/whisper.cpp/build/bin/whisper-cli`. Uses GGML models (e.g., `ggml-base.bin`, `ggml-small.bin`).
2. **Piper TTS** — installed via `pip install piper-tts`. Usage: `from piper import PiperVoice`. Requires downloading `.onnx` voice model files separately.
3. **Ollama** — installed via official install script. Runs as `ollama serve` in background. Models pulled via `ollama pull gemma3:4b`. API at `http://localhost:11434`.
4. **espeak-ng** — installed via `pkg install espeak`. Fallback TTS. Lower quality but instant and reliable.
5. **FFmpeg** — installed via `pkg install ffmpeg`. For audio format conversion (wav ↔ ogg ↔ mp3, sample rate conversion).

---

## Architecture overview

```
┌─────────────────────────────────────────────────────┐
│                  ANDROID PHONE (Termux)              │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Mic/File │───►│ whisper  │───►│ Ollama   │       │
│  │ (audio)  │    │ .cpp     │    │ (LLM)    │       │
│  └──────────┘    │ [STT]    │    │ [TRANS-  │       │
│                  │          │    │  LATE]   │       │
│                  └────┬─────┘    └────┬─────┘       │
│                       │               │              │
│                  source text     translated text     │
│                       │               │              │
│                       │         ┌─────▼─────┐       │
│                       │         │ Piper TTS │       │
│                       │         │ [SPEAK]   │       │
│                       │         └─────┬─────┘       │
│                       │               │              │
│                       ▼               ▼              │
│                  ┌─────────────────────────┐        │
│                  │   Terminal Output /      │        │
│                  │   Audio Playback /       │        │
│                  │   Telegram Bot           │        │
│                  └─────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

### Pipeline in detail

```
INPUT (one of):
  a) Microphone recording via `termux-microphone-record` or `arecord`
  b) Audio file passed as CLI argument
  c) Telegram voice message (if bot mode enabled)

STEP 1 — STT (Speech-to-Text):
  Tool: whisper.cpp (whisper-cli binary, compiled locally)
  Input: WAV file (16kHz, mono, 16-bit PCM) — convert with ffmpeg if needed
  Output: Plain text + detected source language code (e.g., "ru", "en", "de")
  Model: ggml-base.bin (default) or ggml-small.bin (better quality, slower)

STEP 2 — Translation:
  Tool: Ollama HTTP API (localhost:11434)
  Model: gemma3:4b (default, good quality/speed balance on mobile)
         OR qwen2.5:3b (alternative, good at translation)
  Input: Source text + source language + target language
  Output: Translated text only (no explanations, no commentary)
  System prompt: see TRANSLATION_PROMPT below

STEP 3 — TTS (Text-to-Speech):
  Tool: Piper TTS (via Python piper-tts library)
  Input: Translated text + target language voice model
  Output: WAV audio file
  Fallback: espeak-ng if Piper voice model not available for target language

STEP 4 — Output:
  a) Play audio via `play` (sox) or `termux-media-player`
  b) Save to file
  c) Send as Telegram voice message (bot mode)
```

---

## Project structure

```
termux-voice-translator/
├── CLAUDE.md                    ← this file (project spec for Claude Code)
├── README.md                    ← user-facing documentation (English)
├── LICENSE                      ← MIT license
├── install.sh                   ← one-command installer for Termux
├── requirements.txt             ← Python dependencies
├── config.yaml                  ← user configuration (languages, models, paths)
├── src/
│   ├── __init__.py
│   ├── main.py                  ← entry point (CLI + dispatcher)
│   ├── stt.py                   ← Speech-to-Text module (whisper.cpp wrapper)
│   ├── translator.py            ← Translation module (Ollama wrapper)
│   ├── tts.py                   ← Text-to-Speech module (Piper + espeak fallback)
│   ├── audio.py                 ← Audio utilities (recording, conversion, playback)
│   ├── config.py                ← Configuration loader (YAML)
│   ├── pipeline.py              ← Main translation pipeline (orchestrates all steps)
│   └── telegram_bot.py          ← Optional Telegram bot interface
├── models/                      ← downloaded AI models (gitignored)
│   ├── whisper/                 ← whisper.cpp GGML models
│   └── piper/                   ← Piper .onnx voice models + .json configs
├── tests/
│   ├── test_stt.py
│   ├── test_translator.py
│   ├── test_tts.py
│   ├── test_pipeline.py
│   └── test_audio.py
└── .github/
    └── workflows/
        └── test.yml             ← GitHub Actions CI (lint + test on push)
```

---

## Detailed module specifications

### `src/config.py` — Configuration

Load configuration from `config.yaml` with sane defaults. Use `pydantic` for validation if available, otherwise plain dataclasses.

```yaml
# config.yaml — default configuration
stt:
  engine: whisper-cpp
  whisper_binary: ~/whisper.cpp/build/bin/whisper-cli
  whisper_model: ~/whisper.cpp/models/ggml-base.bin
  language: auto  # "auto" = auto-detect, or force "ru", "en", "de", etc.

translation:
  engine: ollama
  ollama_url: http://localhost:11434
  model: gemma3:4b
  # Alternative models (user can change):
  # model: qwen2.5:3b
  # model: llama3.2:3b
  timeout: 120  # seconds — translation can be slow on mobile

tts:
  engine: piper  # "piper" or "espeak"
  piper_voices_dir: ~/voice-translator/models/piper
  espeak_fallback: true  # use espeak-ng if Piper voice not available
  voices:
    de: de_DE-thorsten-high.onnx
    en: en_US-lessac-medium.onnx
    ru: ru_RU-denis-medium.onnx
    es: es_ES-sharvard-medium.onnx
    fr: fr_FR-siwis-medium.onnx
    lv: # no Piper voice — will use espeak
  speed: 1.0

audio:
  sample_rate: 16000
  format: wav
  record_command: termux-microphone-record  # or "arecord" if available
  play_command: play  # sox "play" command, or "termux-media-player"
  max_duration: 60  # max recording duration in seconds

telegram:
  enabled: false
  bot_token: ""  # set in .env or here
  allowed_users: []  # empty = allow all

general:
  default_source: auto  # auto-detect source language
  default_target: de    # translate TO German by default
  show_source_text: true
  show_translated_text: true
  log_level: INFO
```

### `src/stt.py` — Speech-to-Text module

Wraps whisper.cpp CLI binary. This module does NOT use Python whisper or faster-whisper. It calls the compiled `whisper-cli` binary via subprocess.

```python
class WhisperSTT:
    """
    Speech-to-Text using locally compiled whisper.cpp.
    
    IMPORTANT: whisper-cli expects WAV input: 16kHz, mono, 16-bit PCM.
    Any other format must be converted via ffmpeg BEFORE passing to whisper.
    
    The binary outputs to stdout. Parse the text output, ignore timing lines.
    Language detection: use --language auto flag. Parse detected language from
    whisper-cli stderr output (line: "auto-detected language: xx").
    """
    
    def __init__(self, binary_path: str, model_path: str):
        # Validate binary exists and is executable
        # Validate model file exists
        pass
    
    async def transcribe(self, audio_path: str, language: str = "auto") -> STTResult:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (any format — will be converted to WAV)
            language: ISO 639-1 code ("ru", "en", "de") or "auto" for detection
        
        Returns:
            STTResult(text=str, language=str, duration_sec=float)
        
        Steps:
            1. Convert to 16kHz mono WAV via ffmpeg (if not already)
            2. Run whisper-cli with --model, --language, --output-txt flags
            3. Parse stdout for transcribed text
            4. Parse stderr for detected language (if language="auto")
            5. Clean up temp files
        
        Error handling:
            - FileNotFoundError if whisper-cli binary missing
            - RuntimeError if whisper-cli returns non-zero exit code
            - TimeoutError if transcription takes >60 seconds
        """
        pass
```

**whisper-cli invocation:**
```bash
whisper-cli \
  --model ~/whisper.cpp/models/ggml-base.bin \
  --language auto \
  --no-timestamps \
  --output-txt \
  --file /tmp/input_16khz.wav
```

**FFmpeg conversion command (ALWAYS run before whisper):**
```bash
ffmpeg -i input.ogg -ar 16000 -ac 1 -c:a pcm_s16le /tmp/input_16khz.wav -y
```

### `src/translator.py` — Translation module

Wraps Ollama HTTP API for text translation.

```python
class OllamaTranslator:
    """
    Translation via local Ollama LLM.
    
    Uses the /api/generate endpoint (NOT /api/chat — simpler for single-turn).
    
    CRITICAL: The system prompt must be extremely strict about output format.
    LLMs tend to add explanations, commentary, or alternative translations.
    The prompt must force the model to output ONLY the translated text.
    """
    
    SYSTEM_PROMPT = """You are a professional translator. Your ONLY job is to translate text.

RULES:
- Output ONLY the translated text. Nothing else.
- Do NOT add explanations, notes, alternatives, or commentary.
- Do NOT add quotation marks around the translation.
- Do NOT say "Here is the translation" or similar.
- Preserve the original tone, style, and register.
- If the input is a single word, output a single word.
- If the input contains profanity, translate it naturally (do not censor).
- If you cannot translate (e.g., gibberish input), output the original text unchanged."""

    async def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """
        Translate text from source to target language.
        
        Args:
            text: Source text to translate
            source_lang: ISO 639-1 code of source language (e.g., "ru")
            target_lang: ISO 639-1 code of target language (e.g., "de")
        
        Returns:
            TranslationResult(text=str, source_lang=str, target_lang=str)
        
        Ollama API call:
            POST http://localhost:11434/api/generate
            {
                "model": "gemma3:4b",
                "prompt": "Translate from Russian to German:\n\nПривет, как дела?",
                "system": SYSTEM_PROMPT,
                "stream": false,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 500
                }
            }
        
        Temperature 0.3: low creativity, high accuracy for translation.
        num_predict 500: cap output length to prevent runaway generation.
        stream false: simpler parsing (wait for complete response).
        
        Error handling:
            - ConnectionError if Ollama not running (suggest: ollama serve &)
            - TimeoutError if translation takes >120 seconds
            - ValueError if Ollama returns empty response
        """
        pass
    
    async def health_check(self) -> bool:
        """Check if Ollama is running and the model is loaded."""
        # GET http://localhost:11434/api/tags
        # Check if configured model is in the list
        pass
```

**Language name mapping (for the prompt):**
```python
LANG_NAMES = {
    "ru": "Russian",
    "en": "English", 
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "lv": "Latvian",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "pl": "Polish",
}
```

### `src/tts.py` — Text-to-Speech module

Uses Piper TTS Python library as primary engine, espeak-ng as fallback.

```python
class PiperTTS:
    """
    Text-to-Speech using Piper (Python library).
    
    CRITICAL TERMUX NOTE:
    The Piper pre-built binary from GitHub DOES NOT WORK on Android.
    It's compiled against glibc, but Termux uses Bionic libc.
    The ELF interpreter /lib/ld-linux-aarch64.so.1 does not exist on Android.
    
    SOLUTION: Use the Python package `piper-tts` (pip install piper-tts).
    This uses ONNX Runtime for inference, which works on Termux.
    
    Voice models: .onnx files downloaded from:
    https://github.com/rhasspy/piper/blob/master/VOICES.md
    Each voice needs TWO files: model.onnx AND model.onnx.json (config).
    """
    
    def __init__(self, voices_dir: str, voice_map: dict):
        # voices_dir: directory containing .onnx voice models
        # voice_map: {"de": "de_DE-thorsten-high.onnx", "en": "en_US-lessac-medium.onnx"}
        # Pre-load voice objects on init for faster synthesis
        pass
    
    async def synthesize(self, text: str, language: str, output_path: str) -> str:
        """
        Convert text to speech audio file.
        
        Args:
            text: Text to speak
            language: ISO 639-1 code — maps to voice via voice_map
            output_path: Where to save the output WAV file
        
        Returns:
            Path to generated WAV file
        
        Piper usage (Python library):
            from piper import PiperVoice
            voice = PiperVoice.load("de_DE-thorsten-high.onnx")
            with wave.open(output_path, "wb") as wav_file:
                voice.synthesize(text, wav_file)
        
        Fallback to espeak-ng if:
            - Piper voice model not found for this language
            - Piper synthesis fails for any reason
        """
        pass


class EspeakTTS:
    """
    Fallback TTS using espeak-ng (installed via pkg install espeak).
    Lower quality but works for any language without downloading models.
    """
    
    async def synthesize(self, text: str, language: str, output_path: str) -> str:
        """
        espeak-ng invocation:
            espeak-ng -v {language} -w {output_path} "{text}"
        
        Language codes for espeak: "ru", "en", "de", "es", "fr", "lv"
        """
        pass
```

**Piper voice model download URLs (for install.sh):**
```
# German (thorsten — high quality male voice)
https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx
https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json

# English (lessac — medium quality female voice)
https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# Russian (denis — medium quality male voice)
https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx
https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json
```

### `src/audio.py` — Audio utilities

```python
class AudioManager:
    """
    Handles recording, format conversion, and playback.
    All operations use subprocess calls to ffmpeg and Termux audio tools.
    """
    
    async def record(self, output_path: str, max_duration: int = 60) -> str:
        """
        Record audio from microphone.
        
        Strategy (in order of preference):
        1. termux-microphone-record (best for Termux, outputs to file)
           Command: termux-microphone-record -f {output_path} -l {max_duration}
           Stop: termux-microphone-record -q
           NOTE: Requires Termux:API add-on installed from F-Droid
           
        2. arecord (if available, needs pulseaudio package)
           Command: arecord -f S16_LE -r 16000 -c 1 -d {max_duration} {output_path}
        
        3. Manual mode: prompt user to provide a file path instead
        
        User interaction:
            Print "🎤 Recording... Press Enter to stop (max {max_duration}s)"
            Wait for Enter keypress OR timeout
            Stop recording
        """
        pass
    
    async def convert_to_wav_16khz(self, input_path: str) -> str:
        """
        Convert any audio file to whisper-compatible WAV.
        Output: 16kHz, mono, 16-bit PCM, WAV format.
        
        Command:
            ffmpeg -i {input_path} -ar 16000 -ac 1 -c:a pcm_s16le {output_path} -y
        
        -y flag: overwrite without asking
        Returns path to converted file (in /tmp or tempfile.mkstemp)
        """
        pass
    
    async def play(self, audio_path: str):
        """
        Play audio file through device speakers.
        
        Strategy (in order):
        1. play (from sox package): play {audio_path}
        2. termux-media-player play {audio_path}
        3. ffplay -nodisp -autoexit {audio_path}
        """
        pass
```

### `src/pipeline.py` — Main translation pipeline

```python
class TranslationPipeline:
    """
    Orchestrates the full voice-to-voice translation pipeline.
    
    This is the core class. It chains: audio → STT → translate → TTS → play.
    
    Design decisions:
    - All steps are async for non-blocking execution
    - Each step logs timing info for performance monitoring
    - Intermediate results (source text, translated text) are printed to terminal
    - Errors in any step abort the pipeline with a clear error message
    - Temp files are cleaned up in a finally block
    """
    
    async def translate_voice(
        self,
        audio_input: str,       # file path or "mic" for live recording
        source_lang: str,       # "auto" or ISO code
        target_lang: str,       # ISO code
        play_result: bool = True,
        save_audio: str = None,  # optional path to save translated audio
    ) -> PipelineResult:
        """
        Full pipeline execution.
        
        Returns PipelineResult with:
            source_text: str
            source_language: str (detected or specified)
            translated_text: str
            target_language: str
            audio_path: str (path to generated audio)
            timings: dict (stt_ms, translate_ms, tts_ms, total_ms)
        
        Terminal output during execution:
            🎤 Recording... Press Enter to stop
            ⏳ Transcribing with Whisper...
            📝 Source (ru): "Привет, как дела?"
            ⏳ Translating to German...
            🔄 Translation (de): "Hallo, wie geht es dir?"
            ⏳ Generating speech...
            🔊 Playing translation...
            ✅ Done! (STT: 2.1s | Translate: 4.3s | TTS: 0.8s | Total: 7.2s)
        """
        pass
```

### `src/main.py` — CLI entry point

```python
"""
CLI interface for the voice translator.

Usage modes:

1. Interactive (default — record from mic, translate, speak):
   python -m src.main
   python -m src.main --to de
   python -m src.main --from ru --to de

2. File input:
   python -m src.main --file recording.ogg --to de
   python -m src.main --file audio.mp3 --from en --to ru

3. Text-only (skip STT, just translate + speak):
   python -m src.main --text "Привет мир" --to de

4. Continuous mode (loop: record → translate → repeat):
   python -m src.main --continuous --to de

5. Telegram bot mode:
   python -m src.main --telegram
"""

import argparse
import asyncio

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-translator",
        description="Offline voice-to-voice translator. Runs 100% locally on Android/Termux."
    )
    parser.add_argument("--from", dest="source_lang", default="auto",
                        help="Source language (ISO code: ru, en, de, ...). Default: auto-detect")
    parser.add_argument("--to", dest="target_lang", default="de",
                        help="Target language (ISO code). Default: de (German)")
    parser.add_argument("--file", "-f", help="Input audio file path (skip recording)")
    parser.add_argument("--text", "-t", help="Input text (skip recording + STT)")
    parser.add_argument("--continuous", "-c", action="store_true",
                        help="Continuous mode: loop record → translate → repeat")
    parser.add_argument("--save", "-s", help="Save translated audio to this path")
    parser.add_argument("--telegram", action="store_true",
                        help="Run as Telegram bot")
    parser.add_argument("--no-play", action="store_true",
                        help="Don't play audio output (just print text)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available TTS voices and exit")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models and exit")
    return parser

async def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Load config
    # Initialize pipeline components
    # Dispatch to appropriate mode (interactive, file, text, continuous, telegram)
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

### `src/telegram_bot.py` — Optional Telegram bot

```python
"""
Telegram bot interface for the translator.

Uses aiogram 3.x (async Telegram bot framework — already familiar to the developer).

Bot commands:
    /start — Welcome message + language selection
    /lang de — Set target language to German
    /lang en — Set target language to English
    /help — Show available commands

Voice message handling:
    1. User sends voice message
    2. Bot downloads .ogg file
    3. Pipeline: STT → detect language → translate to target → TTS
    4. Bot sends reply: text translation + voice message with spoken translation

Text message handling:
    1. User sends text
    2. Bot detects language (via simple heuristic or first word)
    3. Translates to target language
    4. Sends reply: text + voice
"""

# Use aiogram 3.x: pip install aiogram
from aiogram import Bot, Dispatcher, F, types
```

---

## install.sh — One-command installer

The installer must be a SINGLE file that does everything. User experience:

```bash
curl -sSL https://raw.githubusercontent.com/aleksbuss/termux-voice-translator/main/install.sh | bash
```

### What install.sh does (in order):

```bash
#!/data/data/com.termux/files/usr/bin/bash
# Termux Voice Translator — Installer
# Tested on: Termux (F-Droid) on Android 12+, ARM64

set -euo pipefail

INSTALL_DIR="$HOME/voice-translator"
WHISPER_DIR="$INSTALL_DIR/whisper.cpp"
MODELS_DIR="$INSTALL_DIR/models"
PIPER_VOICES_DIR="$MODELS_DIR/piper"
WHISPER_MODELS_DIR="$MODELS_DIR/whisper"

echo "╔══════════════════════════════════════════════╗"
echo "║   TERMUX VOICE TRANSLATOR — INSTALLER        ║"
echo "║   100% Offline · Zero Cloud · Zero Cost       ║"
echo "╚══════════════════════════════════════════════╝"

# Step 1: System packages
# pkg update -y && pkg upgrade -y
# pkg install -y python ffmpeg git cmake make clang wget espeak sox termux-api

# Step 2: Create directory structure
# mkdir -p $INSTALL_DIR/src $MODELS_DIR $PIPER_VOICES_DIR $WHISPER_MODELS_DIR

# Step 3: Clone project repo
# git clone https://github.com/aleksbuss/termux-voice-translator $INSTALL_DIR

# Step 4: Python dependencies
# pip install --break-system-packages piper-tts aiogram pyyaml httpx

# Step 5: Compile whisper.cpp from source
# git clone https://github.com/ggml-org/whisper.cpp $WHISPER_DIR
# cd $WHISPER_DIR && mkdir -p build && cd build
# cmake .. && make -j$(nproc)
# VERIFY: test -f $WHISPER_DIR/build/bin/whisper-cli || { echo "whisper build FAILED"; exit 1; }

# Step 6: Download whisper model (ggml-base.bin — 142MB, good speed/quality balance)
# wget -q --show-progress -O $WHISPER_MODELS_DIR/ggml-base.bin \
#   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

# Step 7: Download Piper TTS voice models (German + English + Russian)
# For each voice: download both .onnx and .onnx.json files
# wget -q --show-progress -O $PIPER_VOICES_DIR/de_DE-thorsten-high.onnx \
#   https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx
# (repeat for .json, English, Russian)

# Step 8: Check Ollama
# if command -v ollama &>/dev/null; then
#   echo "✅ Ollama found"
# else
#   echo "⚠️  Ollama not found. Installing..."
#   curl -fsSL https://ollama.com/install.sh | sh
# fi

# Step 9: Pull translation model
# ollama pull gemma3:4b

# Step 10: Generate default config.yaml
# (write config.yaml with correct paths based on $INSTALL_DIR)

# Step 11: Create launcher script
# cat > $INSTALL_DIR/translate.sh << 'EOF'
# #!/data/data/com.termux/files/usr/bin/bash
# cd ~/voice-translator
# python -m src.main "$@"
# EOF
# chmod +x $INSTALL_DIR/translate.sh
# ln -sf $INSTALL_DIR/translate.sh $PREFIX/bin/translate

# Step 12: Verify installation
# Test each component independently:
# - whisper-cli --help
# - python -c "from piper import PiperVoice; print('Piper OK')"
# - curl -s http://localhost:11434/api/tags (if ollama serve is running)
# - espeak-ng --version

# Step 13: Print success message with usage instructions
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   ✅ INSTALLATION COMPLETE!                   ║"
echo "║                                               ║"
echo "║   Quick start:                                ║"
echo "║     translate              (record & translate)║"
echo "║     translate --to en      (translate to English)║"
echo "║     translate --file a.ogg (translate a file)  ║"
echo "║     translate --continuous (loop mode)         ║"
echo "║     translate --telegram   (Telegram bot)      ║"
echo "║                                               ║"
echo "║   Before first use, start Ollama:             ║"
echo "║     ollama serve &                            ║"
echo "║                                               ║"
echo "║   Installed components:                       ║"
echo "║     STT:  whisper.cpp (ggml-base model)       ║"
echo "║     LLM:  Ollama + gemma3:4b                  ║"
echo "║     TTS:  Piper (de/en/ru) + espeak fallback  ║"
echo "╚══════════════════════════════════════════════╝"
```

---

## README.md structure

The README must be in English and serve as portfolio piece for potential employers.

### Required sections:
1. **Title + badges** (platform, license, offline status)
2. **One-paragraph description** — what it does, why it's special
3. **Demo GIF/screenshot** — terminal output showing a translation session
4. **Architecture diagram** — ASCII art showing the pipeline (same as above)
5. **Quick start** — 3 commands to install and run
6. **Usage examples** — all CLI modes with example output
7. **Supported languages** — table of languages with STT/TTS/translation support
8. **How it works** — technical explanation of each pipeline stage
9. **Performance** — expected speed on typical Android devices
10. **Configuration** — how to customize (config.yaml reference)
11. **Telegram bot setup** — how to enable bot mode
12. **Troubleshooting** — common issues and fixes
13. **Why this project?** — brief personal note about building AI on edge devices
14. **License** — MIT

---

## Testing strategy

### Unit tests (pytest):

```python
# test_stt.py
# - Test WAV conversion (mock ffmpeg call, verify arguments)
# - Test whisper-cli output parsing (provide sample stdout, verify extracted text)
# - Test language detection parsing from stderr

# test_translator.py  
# - Test Ollama API request construction (verify JSON body format)
# - Test response parsing (extract text from Ollama response)
# - Test system prompt inclusion
# - Test empty response handling

# test_tts.py
# - Test Piper voice loading (mock PiperVoice)
# - Test espeak fallback trigger (when Piper voice missing)
# - Test output file creation

# test_pipeline.py
# - Test full pipeline with mocked components
# - Test error propagation (STT fails → pipeline aborts with clear message)
# - Test timing measurement

# test_audio.py
# - Test format detection (wav, ogg, mp3)
# - Test ffmpeg command construction
```

### GitHub Actions CI:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-asyncio
      - run: pytest tests/ -v
```

Note: CI tests use mocked external dependencies (whisper-cli, Ollama, Piper). Real integration testing happens on the actual Termux device.

---

## Python dependencies (requirements.txt)

```
piper-tts>=1.2.0
aiogram>=3.4.0
pyyaml>=6.0
httpx>=0.27.0
```

Optional (for development):
```
pytest>=8.0
pytest-asyncio>=0.23
```

---

## Code style and conventions

- **Python 3.12+** (Termux default)
- **Async throughout** — all I/O operations use asyncio (subprocess calls use `asyncio.create_subprocess_exec`)
- **Type hints** on all function signatures
- **Docstrings** on all public methods (Google style)
- **Logging** via `logging` module, not `print()` (except for user-facing terminal output)
- **No global state** — all configuration passed via constructor injection
- **Error messages** must be actionable: not "Error occurred" but "Ollama is not running. Start it with: ollama serve &"
- **Temp files** created via `tempfile` module, cleaned up in `finally` blocks
- **Constants** in UPPER_SNAKE_CASE at module level

---

## Edge cases to handle

1. **Ollama not running** → detect, print clear instructions to start it
2. **Model not pulled** → detect, offer to pull it (`ollama pull gemma3:4b`)
3. **No microphone permission** → detect, print `termux-setup-storage` instructions
4. **Audio file is empty or corrupt** → detect before sending to whisper
5. **whisper returns empty text** → (silence or noise) → print "No speech detected"
6. **Translation returns source text unchanged** → may happen if LLM doesn't know the language
7. **Piper voice not available** → seamlessly fall back to espeak-ng
8. **Very long text** → chunk into sentences for TTS (Piper handles long text poorly)
9. **Phone overheating / slow** → log timing, warn if translation takes >60s
10. **Ctrl+C during recording** → graceful shutdown, clean up temp files
11. **Disk space full** → check before downloading models in install.sh
12. **Old Termux from Google Play** → detect and warn (F-Droid version required)