#!/data/data/com.termux/files/usr/bin/bash
# Termux Voice Translator — Installer
# Tested on: Termux (F-Droid) on Android 12+, ARM64
set -euo pipefail

INSTALL_DIR="$HOME/voice-translator"
WHISPER_DIR="$INSTALL_DIR/whisper.cpp"
MODELS_DIR="$INSTALL_DIR/models"
PIPER_VOICES_DIR="$MODELS_DIR/piper"
WHISPER_MODELS_DIR="$MODELS_DIR/whisper"

STEP=0

print_step() {
    STEP=$((STEP + 1))
    echo ""
    echo "[$STEP] $1"
    echo "────────────────────────────────────────"
}

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  Already exists: $dest"
    else
        echo "  Downloading: $(basename "$dest")..."
        wget -q --show-progress -O "$dest" "$url"
    fi
}

echo "╔══════════════════════════════════════════════╗"
echo "║   TERMUX VOICE TRANSLATOR — INSTALLER        ║"
echo "║   100% Offline · Zero Cloud · Zero Cost       ║"
echo "╚══════════════════════════════════════════════╝"

# ── Environment check ─────────────────────────────────────────────
print_step "Checking environment"

ARCH=$(uname -m)
echo "  Architecture: $ARCH"

if [[ "${PREFIX:-}" == *"com.termux"* ]]; then
    echo "  Environment: Termux (OK)"
else
    echo "  WARNING: Not running in Termux. Some features may not work."
    echo "  Continuing anyway (useful for testing)..."
fi

if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo "  WARNING: Expected aarch64/arm64, got $ARCH."
    echo "  whisper.cpp and Piper models are optimized for ARM64."
fi

# ── System packages ───────────────────────────────────────────────
print_step "Installing system packages"

pkg update -y
pkg install -y python ffmpeg git cmake make clang wget espeak sox termux-api

# ── Directory structure ───────────────────────────────────────────
print_step "Creating directory structure"

mkdir -p "$INSTALL_DIR/src"
mkdir -p "$MODELS_DIR"
mkdir -p "$PIPER_VOICES_DIR"
mkdir -p "$WHISPER_MODELS_DIR"

# ── Clone or update project ──────────────────────────────────────
print_step "Setting up project files"

if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  Project already cloned, pulling latest..."
    cd "$INSTALL_DIR" && git pull
else
    if [ -d "$INSTALL_DIR/src" ] && [ -f "$INSTALL_DIR/src/main.py" ]; then
        echo "  Project files already exist (manual install)."
    else
        echo "  Cloning repository..."
        git clone https://github.com/aleksbuss/termux-voice-translator "$INSTALL_DIR" || {
            echo "  WARNING: git clone failed. If you already have the files, that's OK."
        }
    fi
fi

cd "$INSTALL_DIR"

# ── Python dependencies ──────────────────────────────────────────
print_step "Installing Python dependencies"

pip install --break-system-packages piper-tts aiogram pyyaml httpx 2>/dev/null || \
pip install piper-tts aiogram pyyaml httpx

# ── Compile whisper.cpp ──────────────────────────────────────────
print_step "Building whisper.cpp from source"

if [ -f "$WHISPER_DIR/build/bin/whisper-cli" ]; then
    echo "  whisper-cli already built, skipping."
else
    if [ ! -d "$WHISPER_DIR" ]; then
        echo "  Cloning whisper.cpp..."
        git clone https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR"
    fi

    echo "  Compiling (this may take 5-10 minutes)..."
    cd "$WHISPER_DIR"
    mkdir -p build
    cd build
    cmake ..
    make -j"$(nproc)"

    if [ ! -f "$WHISPER_DIR/build/bin/whisper-cli" ]; then
        echo "  ERROR: whisper.cpp build failed!"
        echo "  Try manually: cd $WHISPER_DIR/build && cmake .. && make -j\$(nproc)"
        exit 1
    fi
    echo "  whisper-cli built successfully."
fi

cd "$INSTALL_DIR"

# ── Download Whisper model ───────────────────────────────────────
print_step "Downloading Whisper model (ggml-base.bin, ~142MB)"

download_if_missing \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" \
    "$WHISPER_MODELS_DIR/ggml-base.bin"

# ── Download Piper TTS voices ────────────────────────────────────
print_step "Downloading Piper TTS voice models"

PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

# German (thorsten — high quality)
echo "  German voice..."
download_if_missing "$PIPER_BASE/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx" \
    "$PIPER_VOICES_DIR/de_DE-thorsten-high.onnx"
download_if_missing "$PIPER_BASE/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json" \
    "$PIPER_VOICES_DIR/de_DE-thorsten-high.onnx.json"

# English (lessac — medium quality)
echo "  English voice..."
download_if_missing "$PIPER_BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
    "$PIPER_VOICES_DIR/en_US-lessac-medium.onnx"
download_if_missing "$PIPER_BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
    "$PIPER_VOICES_DIR/en_US-lessac-medium.onnx.json"

# Russian (denis — medium quality)
echo "  Russian voice..."
download_if_missing "$PIPER_BASE/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx" \
    "$PIPER_VOICES_DIR/ru_RU-denis-medium.onnx"
download_if_missing "$PIPER_BASE/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json" \
    "$PIPER_VOICES_DIR/ru_RU-denis-medium.onnx.json"

# ── Ollama ────────────────────────────────────────────────────────
print_step "Checking Ollama"

if command -v ollama &>/dev/null; then
    echo "  Ollama found: $(command -v ollama)"
else
    echo "  Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh || {
        echo "  WARNING: Ollama install failed."
        echo "  Install manually: curl -fsSL https://ollama.com/install.sh | sh"
    }
fi

# Try to pull model (only works if ollama is serving)
print_step "Pulling Ollama translation model"

if command -v ollama &>/dev/null; then
    echo "  Attempting to pull gemma3:4b..."
    echo "  (If this fails, run: ollama serve & then ollama pull gemma3:4b)"
    ollama pull gemma3:4b 2>/dev/null || {
        echo "  Could not pull model (Ollama may not be serving)."
        echo "  After installation, run:"
        echo "    ollama serve &"
        echo "    ollama pull gemma3:4b"
    }
else
    echo "  Skipping model pull (Ollama not installed)."
fi

# ── Generate config.yaml ─────────────────────────────────────────
print_step "Generating config.yaml"

cat > "$INSTALL_DIR/config.yaml" << YAML
stt:
  engine: whisper-cpp
  whisper_binary: $WHISPER_DIR/build/bin/whisper-cli
  whisper_model: $WHISPER_MODELS_DIR/ggml-base.bin
  language: auto

translation:
  engine: ollama
  ollama_url: http://localhost:11434
  model: gemma3:4b
  timeout: 120

tts:
  engine: piper
  piper_voices_dir: $PIPER_VOICES_DIR
  espeak_fallback: true
  voices:
    de: de_DE-thorsten-high.onnx
    en: en_US-lessac-medium.onnx
    ru: ru_RU-denis-medium.onnx
    es: es_ES-sharvard-medium.onnx
    fr: fr_FR-siwis-medium.onnx
    lv: null
  speed: 1.0

audio:
  sample_rate: 16000
  format: wav
  record_command: termux-microphone-record
  play_command: play
  max_duration: 60

telegram:
  enabled: false
  bot_token: ""
  allowed_users: []

general:
  default_source: auto
  default_target: de
  show_source_text: true
  show_translated_text: true
  log_level: INFO
YAML

echo "  Config written to $INSTALL_DIR/config.yaml"

# ── Launcher script ──────────────────────────────────────────────
print_step "Creating launcher"

cat > "$INSTALL_DIR/translate.sh" << 'LAUNCHER'
#!/data/data/com.termux/files/usr/bin/bash
cd ~/voice-translator
python -m src.main "$@"
LAUNCHER

chmod +x "$INSTALL_DIR/translate.sh"

# Create symlink in $PREFIX/bin for global access
if [ -n "${PREFIX:-}" ]; then
    ln -sf "$INSTALL_DIR/translate.sh" "$PREFIX/bin/translate"
    echo "  Command 'translate' available globally."
else
    echo "  Launcher at: $INSTALL_DIR/translate.sh"
fi

# ── Verify installation ──────────────────────────────────────────
print_step "Verifying installation"

ERRORS=0

echo -n "  whisper-cli: "
if [ -f "$WHISPER_DIR/build/bin/whisper-cli" ]; then
    echo "OK"
else
    echo "MISSING"
    ERRORS=$((ERRORS + 1))
fi

echo -n "  Whisper model: "
if [ -f "$WHISPER_MODELS_DIR/ggml-base.bin" ]; then
    echo "OK ($(du -h "$WHISPER_MODELS_DIR/ggml-base.bin" | cut -f1))"
else
    echo "MISSING"
    ERRORS=$((ERRORS + 1))
fi

echo -n "  Piper TTS (Python): "
if python -c "from piper import PiperVoice; print('OK')" 2>/dev/null; then
    :
else
    echo "MISSING (pip install piper-tts)"
    ERRORS=$((ERRORS + 1))
fi

echo -n "  espeak-ng: "
if command -v espeak-ng &>/dev/null; then
    echo "OK"
else
    echo "MISSING (pkg install espeak)"
    ERRORS=$((ERRORS + 1))
fi

echo -n "  ffmpeg: "
if command -v ffmpeg &>/dev/null; then
    echo "OK"
else
    echo "MISSING (pkg install ffmpeg)"
    ERRORS=$((ERRORS + 1))
fi

echo -n "  Ollama: "
if command -v ollama &>/dev/null; then
    echo "OK"
else
    echo "MISSING"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "  $ERRORS component(s) have issues. Check above."
fi

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   INSTALLATION COMPLETE!                      ║"
echo "║                                               ║"
echo "║   Quick start:                                ║"
echo "║     translate              (record & translate)║"
echo "║     translate --to en      (translate to Eng.) ║"
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
