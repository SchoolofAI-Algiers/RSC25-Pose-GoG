from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories

DATA_RAW = PROJECT_ROOT / "data" / "nturgb_raw" / "nturgb+d_skeletons"
DATA_STATS = PROJECT_ROOT / "data" / "ntu" / "statistics"

RAW_SKELS = PROJECT_ROOT / "data" / "ntu" / "raw_data"
DENOISED_SKELS = PROJECT_ROOT / "data" / "ntu" / "denoised_data"

DATA_PROCESSED = PROJECT_ROOT / "data" / "ntu"

NTU_FINAL_PROCESSED_DATA = PROJECT_ROOT / "processed_data" / "ntu60"

# Results directories
RESULTS = PROJECT_ROOT / "results"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_TABLES = RESULTS / "tables"

# Ensure directories exist
for path in [DATA_RAW, RAW_SKELS, DENOISED_SKELS, DATA_PROCESSED]:
    path.mkdir(parents=True, exist_ok=True)