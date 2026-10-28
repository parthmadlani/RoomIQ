#!/bin/zsh
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  runner=.venv/bin/python
else
  runner=python3
fi
if ! "$runner" -c "import sklearn, pandas, numpy, joblib" 2>/dev/null; then
  echo "RoomIQ needs its Python packages. Set up the project environment once (README section 7):"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  read "?Press Enter to close."
  exit 1
fi
"$runner" src/dashboard_server.py --open
if [[ $? -ne 0 ]]; then
  echo "Dashboard could not start. See README.md for setup, or use another port with --port 8766."
  read "?Press Enter to close."
fi
