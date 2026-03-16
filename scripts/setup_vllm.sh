#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# NyayaMitra — vLLM Setup Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sets up vLLM inference server with Llama 3.1 8B (development model)
# for local RAG pipeline testing.
#
# Supports:
#   - NVIDIA GPU (Linux/WSL) — native vLLM with CUDA
#   - Apple Silicon (macOS)  — llama.cpp via llama-cpp-python (vLLM
#                               doesn't support MPS yet)
#   - CPU fallback           — slower but works anywhere
#
# Usage:
#   chmod +x scripts/setup_vllm.sh
#   ./scripts/setup_vllm.sh              # auto-detect platform
#   ./scripts/setup_vllm.sh --gpu        # force GPU mode
#   ./scripts/setup_vllm.sh --cpu        # force CPU mode
#   ./scripts/setup_vllm.sh --apple      # force Apple Silicon mode
#   ./scripts/setup_vllm.sh --check      # just check if server is running
#   ./scripts/setup_vllm.sh --stop       # stop running server
#
# After setup, the server runs on http://localhost:8000 with an
# OpenAI-compatible /v1/chat/completions endpoint.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${VLLM_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
# For Apple Silicon, we use a GGUF quantized model
GGUF_MODEL="${GGUF_MODEL:-bartowski/Meta-Llama-3.1-8B-Instruct-GGUF}"
GGUF_FILE="${GGUF_FILE:-Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
GPU_MEMORY="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
PID_FILE="/tmp/nyayamitra-llm-server.pid"

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ─── Platform Detection ─────────────────────────────────────────────────────
detect_platform() {
    if [[ "${FORCE_MODE:-}" == "gpu" ]]; then
        echo "gpu"
    elif [[ "${FORCE_MODE:-}" == "cpu" ]]; then
        echo "cpu"
    elif [[ "${FORCE_MODE:-}" == "apple" ]]; then
        echo "apple"
    elif [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        echo "apple"
    elif command -v nvidia-smi &>/dev/null; then
        echo "gpu"
    else
        echo "cpu"
    fi
}

# ─── Health Check ────────────────────────────────────────────────────────────
check_server() {
    local url="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
    if curl -s --max-time 5 "$url" | grep -q "id"; then
        return 0
    fi
    return 1
}

# ─── Stop Server ─────────────────────────────────────────────────────────────
stop_server() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping LLM server (PID: $pid)..."
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            ok "Server stopped."
        else
            warn "PID $pid not running."
        fi
        rm -f "$PID_FILE"
    else
        warn "No PID file found. Checking for running processes..."
        pkill -f "vllm.entrypoints\|llama_cpp.server\|llama-cpp-python" 2>/dev/null && ok "Stopped." || warn "No server found."
    fi
}

# ─── GPU Setup (NVIDIA + vLLM) ──────────────────────────────────────────────
setup_gpu() {
    info "Setting up vLLM with NVIDIA GPU..."

    # Check CUDA
    if ! nvidia-smi &>/dev/null; then
        error "nvidia-smi not found. Install NVIDIA drivers first."
        exit 1
    fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    info "GPU: $gpu_info"

    # Install vLLM
    if ! python -c "import vllm" 2>/dev/null; then
        info "Installing vLLM..."
        pip install vllm --break-system-packages
    fi
    ok "vLLM installed."

    # Check HuggingFace token for gated models
    if [[ -z "${HF_TOKEN:-}" ]]; then
        warn "HF_TOKEN not set. Llama 3.1 is a gated model — you need to:"
        warn "  1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
        warn "  2. Set HF_TOKEN in your .env or environment"
        warn "  export HF_TOKEN=hf_your_token_here"
    fi

    info "Starting vLLM server..."
    info "Model: $MODEL_NAME"
    info "Port: $VLLM_PORT"
    info "GPU Memory: ${GPU_MEMORY}"
    info "Max Context: ${MAX_MODEL_LEN} tokens"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --host "0.0.0.0" \
        --port "$VLLM_PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY" \
        --dtype auto \
        --trust-remote-code &

    echo $! > "$PID_FILE"
    info "Server PID: $(cat $PID_FILE)"
}

# ─── Apple Silicon Setup (llama.cpp) ────────────────────────────────────────
setup_apple() {
    info "Setting up llama-cpp-python for Apple Silicon..."

    # Install llama-cpp-python with Metal support
    if ! python -c "import llama_cpp" 2>/dev/null; then
        info "Installing llama-cpp-python with Metal acceleration..."
        CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python[server] --break-system-packages
    fi
    ok "llama-cpp-python installed with Metal support."

    # Download GGUF model
    local model_dir="$HOME/.cache/nyayamitra/models"
    local model_path="$model_dir/$GGUF_FILE"

    if [[ ! -f "$model_path" ]]; then
        info "Downloading quantized model: $GGUF_MODEL / $GGUF_FILE"
        info "This may take 5-10 minutes on first run..."
        mkdir -p "$model_dir"

        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download "$GGUF_MODEL" "$GGUF_FILE" \
                --local-dir "$model_dir" --local-dir-use-symlinks False
        else
            info "Installing huggingface-hub for model download..."
            pip install huggingface-hub --break-system-packages
            huggingface-cli download "$GGUF_MODEL" "$GGUF_FILE" \
                --local-dir "$model_dir" --local-dir-use-symlinks False
        fi
    fi
    ok "Model ready: $model_path"

    info "Starting llama.cpp server with Metal acceleration..."
    info "Model: $GGUF_FILE"
    info "Port: $VLLM_PORT"
    info "Context: $MAX_MODEL_LEN tokens"

    python -m llama_cpp.server \
        --model "$model_path" \
        --host "0.0.0.0" \
        --port "$VLLM_PORT" \
        --n_ctx "$MAX_MODEL_LEN" \
        --n_gpu_layers -1 \
        --chat_format llama-3 &

    echo $! > "$PID_FILE"
    info "Server PID: $(cat $PID_FILE)"
}

# ─── CPU Setup (llama.cpp without GPU) ──────────────────────────────────────
setup_cpu() {
    info "Setting up llama-cpp-python (CPU mode)..."
    warn "CPU inference is slow. Consider using --gpu or --apple for better performance."

    if ! python -c "import llama_cpp" 2>/dev/null; then
        info "Installing llama-cpp-python..."
        pip install llama-cpp-python[server] --break-system-packages
    fi
    ok "llama-cpp-python installed."

    local model_dir="$HOME/.cache/nyayamitra/models"
    local model_path="$model_dir/$GGUF_FILE"

    if [[ ! -f "$model_path" ]]; then
        info "Downloading quantized model: $GGUF_MODEL / $GGUF_FILE"
        mkdir -p "$model_dir"
        if ! command -v huggingface-cli &>/dev/null; then
            pip install huggingface-hub --break-system-packages
        fi
        huggingface-cli download "$GGUF_MODEL" "$GGUF_FILE" \
            --local-dir "$model_dir" --local-dir-use-symlinks False
    fi
    ok "Model ready: $model_path"

    info "Starting llama.cpp server (CPU)..."

    python -m llama_cpp.server \
        --model "$model_path" \
        --host "0.0.0.0" \
        --port "$VLLM_PORT" \
        --n_ctx "$MAX_MODEL_LEN" \
        --n_gpu_layers 0 \
        --chat_format llama-3 &

    echo $! > "$PID_FILE"
    info "Server PID: $(cat $PID_FILE)"
}

# ─── Wait for Server ────────────────────────────────────────────────────────
wait_for_server() {
    info "Waiting for server to start..."
    local max_wait=120
    local waited=0

    while [[ $waited -lt $max_wait ]]; do
        if check_server; then
            ok "LLM server is ready at http://${VLLM_HOST}:${VLLM_PORT}"
            echo ""
            info "Test with:"
            echo "  curl http://localhost:${VLLM_PORT}/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"model\": \"default\", \"messages\": [{\"role\": \"user\", \"content\": \"What is Section 302 IPC?\"}]}'"
            echo ""
            info "NyayaMitra will auto-detect this server on the next query."
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        printf "\r  Waiting... %ds / %ds" "$waited" "$max_wait"
    done

    echo ""
    error "Server failed to start within ${max_wait}s."
    error "Check logs above for errors."
    return 1
}

# ─── Fallback: Use OpenAI Instead ───────────────────────────────────────────
suggest_openai_fallback() {
    echo ""
    warn "If you can't run a local model, set an API key in .env instead:"
    echo ""
    echo "  # Option A: OpenAI"
    echo "  OPENAI_API_KEY=sk-..."
    echo "  OPENAI_MODEL=gpt-4o-mini"
    echo ""
    echo "  # Option B: Anthropic"
    echo "  ANTHROPIC_API_KEY=sk-ant-..."
    echo "  ANTHROPIC_MODEL=claude-sonnet-4-20250514"
    echo ""
    echo "  # Option C: Any OpenAI-compatible API (Groq, Together, etc.)"
    echo "  LLM_API_URL=https://api.groq.com/openai/v1"
    echo "  LLM_API_KEY=gsk_..."
    echo "  LLM_MODEL_NAME=llama-3.1-8b-instant"
    echo ""
    info "NyayaMitra auto-detects the provider on startup."
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  NyayaMitra — LLM Server Setup"
    echo "═══════════════════════════════════════════════════"
    echo ""

    # Parse args
    FORCE_MODE=""
    for arg in "$@"; do
        case "$arg" in
            --gpu)    FORCE_MODE="gpu" ;;
            --cpu)    FORCE_MODE="cpu" ;;
            --apple)  FORCE_MODE="apple" ;;
            --check)
                if check_server; then
                    ok "LLM server is running at http://${VLLM_HOST}:${VLLM_PORT}"
                    curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" | python3 -m json.tool 2>/dev/null || true
                else
                    warn "LLM server is NOT running."
                fi
                exit 0
                ;;
            --stop)
                stop_server
                exit 0
                ;;
            --help|-h)
                echo "Usage: $0 [--gpu|--cpu|--apple|--check|--stop]"
                echo ""
                echo "  --gpu     Force NVIDIA GPU mode (vLLM)"
                echo "  --apple   Force Apple Silicon mode (llama.cpp + Metal)"
                echo "  --cpu     Force CPU mode (llama.cpp, slow)"
                echo "  --check   Check if server is running"
                echo "  --stop    Stop running server"
                echo ""
                exit 0
                ;;
        esac
    done

    # Check if already running
    if check_server; then
        ok "LLM server is already running at http://${VLLM_HOST}:${VLLM_PORT}"
        exit 0
    fi

    # Detect platform
    local platform
    platform=$(detect_platform)
    info "Platform detected: $platform"

    case "$platform" in
        gpu)    setup_gpu ;;
        apple)  setup_apple ;;
        cpu)    setup_cpu ;;
        *)
            error "Unknown platform: $platform"
            exit 1
            ;;
    esac

    if ! wait_for_server; then
        suggest_openai_fallback
        exit 1
    fi
}

main "$@"