import sys
from pathlib import Path

# No packaging setup in this repo: put the repository root on sys.path so that
# ``import src.*`` resolves when pytest is launched from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
