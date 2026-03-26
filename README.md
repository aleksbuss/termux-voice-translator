# Termux Voice Translator

![Platform](https://img.shields.io/badge/platform-Android%20(Termux)-green)
![License](https://img.shields.io/badge/license-MIT-blue)
![Offline](https://img.shields.io/badge/network-100%25%20offline-orange)
![Architecture](https://img.shields.io/badge/arch-ARM64%20(aarch64)-lightgrey)

**Fully offline voice-to-voice translator running on Android via Termux.** Speak in one language, hear the translation spoken back. Zero internet. Zero API costs. Zero data leaks. Everything runs locally on your phone: speech recognition (whisper.cpp), translation (Ollama + Gemma), and speech synthesis (Piper TTS).

## Demo

```
$ translate --to de

🎤 Recording... Press Enter to stop (max 60s)
⏳ Transcribing with Whisper...
📝 Source (ru): "Привет, как дела?"
⏳ Translating to German...
🔄 Translation (de): "Hallo, wie geht es dir?"
⏳ Generating speech...
🔊 Playing translation...
✅ Done! (STT: 2.1s | Translate: 4.3s | TTS: 0.8s | Total: 7.2s)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  ANDROID PHONE (Termux)             │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Mic/File │───►│ whisper  │───►│ Ollama   │       │
│  │ (audio)  │    │ .cpp     │    │ (Gemma)  │       │
│  └──────────┘    │ [STT]    │    │ [TRANS-  │       │
│                  │          │    │  LATE]   │       │
│                  └────┬─────┘    └────┬─────┘       │
│                       │               │             │
│                  source text     translated text    │
│                       │               │             │
│                       │         ┌─────▼─────┐       │
│                       │         │ Piper TTS │       │
│                       │         │ [SPEAK]   │       │
│                       │         └─────┬─────┘       │
│                       │               │             │
│                       ▼               ▼             │
│                  ┌─────────────────────────┐        │
│                  │   Terminal / Playback /  │       │
│                  │   Telegram Bot           │       │
│                  └─────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

## Quick Start

**Prerequisites:** [Termux](https://f-droid.org/en/packages/com.termux/) from F-Droid (NOT Google Play) on an ARM64 Android device.

```bash
# 1. Install everything (packages, models, dependencies)
curl -sSL https://raw.githubusercontent.com/aleksbuss/termux-voice-translator/main/install.sh | bash

# 2. Start Ollama (needed for translation)
ollama serve &

# 3. Translate!
translate --to de
```

## Usage

```bash
# Interactive — record from mic, translate, speak
translate
translate --to en
translate --from ru --to de

# Translate an audio file
translate --file recording.ogg --to de
translate --file voice.mp3 --from en --to ru

# Text mode — skip speech recognition
translate --text "Привет мир" --to de

# Continuous — loop until Ctrl+C
translate --continuous --to de

# Save translated audio
translate --file input.ogg --to en --save output.wav

# Text only — no audio playback
translate --text "Hello world" --to ru --no-play

# Telegram bot mode
translate --telegram

# Info
translate --list-voices
translate --list-models
```

## Supported Languages

| Language    | Code | STT (Whisper) | Translation | TTS (Piper) | TTS (espeak) |
|-------------|------|:---:|:---:|:---:|:---:|
| Russian     | `ru` | ✅ | ✅ | ✅ | ✅ |
| English     | `en` | ✅ | ✅ | ✅ | ✅ |
| German      | `de` | ✅ | ✅ | ✅ | ✅ |
| Spanish     | `es` | ✅ | ✅ | ✅ | ✅ |
| French      | `fr` | ✅ | ✅ | ✅ | ✅ |
| Latvian     | `lv` | ✅ | ✅ | ❌ | ✅ |
| Italian     | `it` | ✅ | ✅ | — | ✅ |
| Portuguese  | `pt` | ✅ | ✅ | — | ✅ |
| Ukrainian   | `uk` | ✅ | ✅ | — | ✅ |
| Polish      | `pl` | ✅ | ✅ | — | ✅ |
| Chinese     | `zh` | ✅ | ✅ | — | ✅ |
| Japanese    | `ja` | ✅ | ✅ | — | ✅ |
| Korean      | `ko` | ✅ | ✅ | — | ✅ |
| Arabic      | `ar` | ✅ | ✅ | — | ✅ |
| Turkish     | `tr` | ✅ | ✅ | — | ✅ |

Whisper supports 99 languages for recognition. Ollama can translate between any language pair. Piper voices must be downloaded separately; espeak-ng covers everything else as a fallback.

## How It Works

### Step 1 — Speech-to-Text (whisper.cpp)

Audio is converted to 16kHz mono WAV via ffmpeg, then passed to a locally compiled `whisper-cli` binary. The `ggml-base.bin` model (142MB) provides a good balance of speed and accuracy. Language is auto-detected or can be specified.

### Step 2 — Translation (Ollama + Gemma 3 4B)

Recognized text is sent to a local Ollama instance running `gemma3:4b` (2.3GB). A strict system prompt ensures the model outputs only the translated text — no explanations, no commentary. Temperature 0.3 keeps translations accurate.

### Step 3 — Text-to-Speech (Piper TTS)

Translated text is synthesized to speech using Piper's ONNX-based Python library. High-quality neural voices are available for German, English, and Russian. For other languages, espeak-ng provides instant (if lower-quality) fallback. Long text is automatically chunked into sentences for better synthesis.

### Step 4 — Output

The generated audio is played through the device speakers via sox, termux-media-player, or ffplay. In Telegram bot mode, audio is sent as a voice message.

## Performance

Expected speed on a typical modern Android phone (Snapdragon 8 Gen 1 / Dimensity 9000):

| Stage | Time | Notes |
|-------|------|-------|
| STT (whisper-cli, base model) | 2-5s | For 5-10s audio |
| Translation (gemma3:4b) | 3-8s | Depends on text length |
| TTS (Piper) | 0.5-2s | Near-instant for short phrases |
| **Total** | **6-15s** | End-to-end for typical utterance |

Older or budget devices will be slower. Consider using `ggml-tiny.bin` (75MB) for faster STT or `qwen2.5:3b` for faster translation.

## Configuration

Edit `~/voice-translator/config.yaml`:

```yaml
stt:
  whisper_model: ~/voice-translator/models/whisper/ggml-base.bin
  language: auto          # or force: ru, en, de...

translation:
  model: gemma3:4b        # alternatives: qwen2.5:3b, llama3.2:3b
  timeout: 120

tts:
  engine: piper            # or "espeak"
  espeak_fallback: true

general:
  default_target: de       # default translation target language
```

## Telegram Bot Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) and get the token
2. Add the token to `config.yaml`:
   ```yaml
   telegram:
     enabled: true
     bot_token: "123456:ABC..."
     allowed_users: []     # empty = allow all, or [123456789]
   ```
3. Start Ollama and the bot:
   ```bash
   ollama serve &
   translate --telegram
   ```

Send a voice message or text to the bot — it replies with translation + voice.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ollama: command not found` | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Translation hangs | Start Ollama: `ollama serve &` |
| Model not found | `ollama pull gemma3:4b` |
| No microphone | Install Termux:API from F-Droid |
| Empty transcription | Check audio isn't silence; try `ggml-small.bin` for better accuracy |
| Piper voice missing | Download from [Piper voices](https://github.com/rhasspy/piper/blob/master/VOICES.md) |
| `ELF interpreter` error | You're using a pre-built binary. Use `pip install piper-tts` instead |
| Termux from Google Play | Uninstall, reinstall from F-Droid |
| Out of storage | Models need ~3GB total. Free space with `pkg clean` |

## Why This Project?

Running a full AI translation pipeline — speech recognition, LLM translation, and neural TTS — entirely on a phone with zero cloud dependency demonstrates what's possible with edge AI today. No subscription, no API keys, no data leaving your device. Just your voice, your phone, and open-source models.

## License

MIT — see [LICENSE](LICENSE).
