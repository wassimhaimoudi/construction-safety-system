#!/usr/bin/env bash
# ──────────────────────────────────────────────────
#  AI Construction Site Safety Monitor — Web Server
#  Usage:  ./run.sh
# ──────────────────────────────────────────────────

set -e

echo ""
echo "🏗️  Safety Monitor — Web Server Setup"
echo "──────────────────────────────────────"

# Check model files
if [ ! -f "vew.pt" ]; then
  echo "❌  vew.pt not found — place it in this folder"
  exit 1
fi
if [ ! -f "scafandri.pt" ]; then
  echo "❌  scafandri.pt not found — place it in this folder"
  exit 1
fi
echo "✅  vew.pt found"
echo "✅  scafandri.pt found"

# Install dependencies if needed
echo ""
echo "📦  Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "🚀  Starting server → http://localhost:5000"
echo "    Landing page : http://localhost:5000/"
echo "    Analysis app : http://localhost:5000/monitor"
echo ""

python app.py
