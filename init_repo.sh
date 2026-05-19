#!/usr/bin/env bash
# init_repo.sh — create a GitHub repo and push the project.
#
# Two paths:
#   1. If you have the GitHub CLI (`gh`) installed, this script does it all in one go.
#      Install: brew install gh   (macOS)   |   https://cli.github.com/
#   2. If not, the script prints copy-paste instructions for plain git + GitHub web UI.

set -e

REPO_NAME="${1:-xnarrative}"
VISIBILITY="${2:-private}"   # change to "public" if you want it public

echo "→ initializing git repo for: $REPO_NAME (visibility: $VISIBILITY)"

# Initialize git locally
if [ ! -d ".git" ]; then
    git init -b main
    echo "  ✓ git initialized"
else
    echo "  · git already initialized"
fi

git add .
git status --short

echo
read -p "Commit and push these files? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

git commit -m "Initial commit: xnarrative pipeline + Streamlit UI"

if command -v gh &> /dev/null; then
    # Use GitHub CLI
    echo "→ creating $VISIBILITY GitHub repo via gh CLI..."
    gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
    echo
    echo "✓ Done. Repo URL:"
    gh repo view --json url -q .url
else
    # Fallback: print instructions
    echo
    echo "─────────────────────────────────────────────────────────────"
    echo "  gh CLI not found. Finish setup manually:"
    echo "─────────────────────────────────────────────────────────────"
    echo
    echo "  1. Go to https://github.com/new"
    echo "  2. Repo name: $REPO_NAME    Visibility: $VISIBILITY"
    echo "  3. Do NOT initialize with README, .gitignore, or license"
    echo "     (we already have those)"
    echo "  4. Click 'Create repository'"
    echo "  5. Then copy-paste these commands, replacing YOUR_USERNAME:"
    echo
    echo "     git remote add origin git@github.com:YOUR_USERNAME/$REPO_NAME.git"
    echo "     git branch -M main"
    echo "     git push -u origin main"
    echo
    echo "  (If you use HTTPS instead of SSH:"
    echo "     git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git)"
fi
