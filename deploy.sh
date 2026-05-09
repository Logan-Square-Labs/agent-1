#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

REPO_ID="agent-1"
REPO_ORG="Logan-Square-Labs"
REMOTE_DIR="~/$REPO_ID"
ENV_FILE=".env"

parse_args() {
    HOST=${1:?Usage: ./deploy.sh user@host [--env /path/to/.env]}
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env) ENV_FILE=$2; shift 2 ;;
            *) echo "Unknown arg: $1"; exit 1 ;;
        esac
    done
}

has_ssh_access() {
    set +e
    timeout 5s git ls-remote --heads "git@github.com:${REPO_ORG}/${REPO_ID}.git" >/dev/null 2>&1
    rc=$?
    set -e
    return $rc
}

copy_env() {
    log_info "Copying secrets to remote..."
    ssh "$HOST" "cat > $REMOTE_DIR/.env" < "$ENV_FILE"
}

main() {
    parse_args "$@"

    [[ -f "$ENV_FILE" ]] || { echo "Env file not found: $ENV_FILE"; exit 1; }

    log_info "Determining clone method (SSH vs HTTPS)..."
    if has_ssh_access; then
        log_info "SSH access to GitHub available."
        CLONE_URL="git@github.com:${REPO_ORG}/${REPO_ID}.git"
    else
        log_warn "SSH auth unavailable, falling back to HTTPS."
        CLONE_URL="https://github.com/${REPO_ORG}/${REPO_ID}.git"
    fi

    log_info "Setting up $HOST..."
    ssh -A "$HOST" "
        if [[ -d $REMOTE_DIR ]]; then
            echo '[INFO] Repo exists, pulling latest...'
            cd $REMOTE_DIR && git pull
        else
            echo '[INFO] Cloning repo...'
            git clone $CLONE_URL $REMOTE_DIR
        fi
        cd $REMOTE_DIR && bash scripts/setup.sh
    "

    copy_env

    log_info "Done. Connect with: ssh $HOST"
    log_info "Then: cd $REPO_ID && tmux new -s train"
}

main "$@"
