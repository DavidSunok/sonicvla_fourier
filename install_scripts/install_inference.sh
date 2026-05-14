#!/usr/bin/env bash
# install_inference.sh
# Sets up the .venv_inference venv for running VLA inference with
# Isaac-GR00T PolicyClient against a remote or local policy server.
#
# Installs gear_sonic[inference] which pulls in the Isaac-GR00T library,
# PyZMQ, msgpack, Pinocchio, and other inference dependencies.
#
# Usage:  bash install_scripts/install_inference.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 0. System dependencies ────────────────────────────────────────────────────
ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

# ── 1. Ensure uv is installed and available ──────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install a uv-managed Python 3.10 (includes dev headers / Python.h) ────
echo "[INFO] Installing uv-managed Python 3.10 (includes development headers) …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv (if any) ──────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_inference (if present) …"
rm -rf .venv_inference

# ── 4. Create venv & install inference extra ─────────────────────────────────
echo "[INFO] Creating .venv_inference with uv-managed Python 3.10 …"
uv venv .venv_inference --python "$MANAGED_PY" --prompt gear_sonic_inference
# shellcheck disable=SC1091
source .venv_inference/bin/activate
echo "[INFO] Installing gear_sonic[inference] (this may take a few minutes) …"
uv pip install -e "gear_sonic[inference]"

# ── 4b. Install Isaac-GR00T from local clone ────────────────────────────────
GR00T_LOCAL="/mnt/mydrive/data/model_base/Isaac-GR00T"
if [ -d "$GR00T_LOCAL" ]; then
    echo "[INFO] Installing Isaac-GR00T from local clone…"
    uv pip install --no-deps -e "$GR00T_LOCAL"
else
    echo "[INFO] Local Isaac-GR00T not found, installing from git (may be slow)…"
    uv pip install --no-deps "gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T.git"
fi

# ── 5. Install Fourier hand SDK (optional, for --hand-type fourier inference) ─
echo ""
echo "[INFO] Checking for Fourier hand SDK (dexhandpy)…"
TELEOP_VENV="$REPO_ROOT/../pipeline/GR00T-WholeBodyControl/.venv_teleop"
SDK_SRC="$TELEOP_VENV/lib/python3.10/site-packages/dexhandpy"
LIB_SRC="$TELEOP_VENV/lib/python3.10/site-packages/fdexhand"

if [ -d "$SDK_SRC" ]; then
    echo "[INFO] Copying dexhandpy from .venv_teleop…"
    cp -r "$SDK_SRC" .venv_inference/lib/python3.10/site-packages/
    if [ -d "$LIB_SRC" ]; then
        cp -r "$LIB_SRC" .venv_inference/lib/python3.10/site-packages/
    fi
    echo "[OK] Fourier hand SDK installed (for --hand-type fourier inference)"
else
    echo "[SKIP] dexhandpy not found at $SDK_SRC"
    echo "       If you need Fourier hand support, install dexhandpy manually."
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!  Activate the venv with:"
echo ""
echo "    source .venv_inference/bin/activate"
echo ""
echo "  You should see (gear_sonic_inference) in your prompt."
echo ""
echo "  Then run VLA inference with:"
echo "    python gear_sonic/scripts/run_vla_inference.py --help"
echo "══════════════════════════════════════════════════════════════"
