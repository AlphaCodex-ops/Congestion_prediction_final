#!/usr/bin/env bash
set -euo pipefail

python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[test,plots]"
.venv/bin/python -m congestion_project.cli doctor
echo "Setup complete. Try: .venv/bin/python -m congestion_project.cli demo"
