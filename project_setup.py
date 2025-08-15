# project_setup.py
from pathlib import Path

def find_project_root():
    """Walk upwards until we find the project root (identified by .git or pyproject.toml)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root. Add .git or pyproject.toml at the root.")

PROJECT_ROOT = find_project_root()

# ---- DATA DIRECTORIES ----
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# ---- RESULTS DIRECTORIES ----
RESULTS = PROJECT_ROOT / "results"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_TABLES = RESULTS / "tables"

# Ensure directories exist
for path in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, RESULTS, RESULTS_FIGURES, RESULTS_TABLES]:
    path.mkdir(parents=True, exist_ok=True)

# ---- CONVENIENCE FUNCTIONS ----
def data_path(*parts, stage="raw"):
    """Get a path inside a data stage: raw, interim, processed."""
    if stage == "raw":
        return DATA_RAW.joinpath(*parts)
    elif stage == "interim":
        return DATA_INTERIM.joinpath(*parts)
    elif stage == "processed":
        return DATA_PROCESSED.joinpath(*parts)
    else:
        raise ValueError("stage must be 'raw', 'interim', or 'processed'")

def results_path(*parts, category=None):
    """Get a path inside results: optional category (figures/tables/custom)."""
    if category == "figures":
        base = RESULTS_FIGURES
    elif category == "tables":
        base = RESULTS_TABLES
    else:
        base = RESULTS
    return base.joinpath(*parts)
