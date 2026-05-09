#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

main() {
    if ! command -v sudo &>/dev/null; then
        apt update && apt install -y sudo
    fi

    log_info "Installing base packages..."
    sudo apt update && sudo apt install -y build-essential curl git tmux htop nvtop

    if ! command -v uv &>/dev/null; then
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source "$HOME/.local/bin/env"
    else
        log_warn "uv already installed, skipping."
    fi

    log_info "Syncing virtual environment..."
    uv sync

    log_info "Setup complete!"
}

main
