from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

DS003838_RAW = DATA_DIR / "ds003838" / "raw"
DS003838_SLIM = DATA_DIR / "ds003838" / "slim"
DS003838_DERIVED = DATA_DIR / "ds003838" / "derived"
DS003838_RESULTS = RESULTS_DIR / "ds003838"
