import os
from pathlib import Path
from dataclasses import dataclass

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input data paths (checking both root and subfolder)
def get_data_file(filename: str) -> Path:
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "AI Intern Case Study" / filename,
        BASE_DIR.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback to subfolder
    return BASE_DIR / "AI Intern Case Study" / filename

@dataclass
class PipelineConfig:
    shipment_records_path: Path = get_data_file("shipment_records.csv")
    context_notes_path: Path = get_data_file("context_notes.csv")
    sample_output_path: Path = get_data_file("sample_output_format_v2.csv")
    output_path: Path = BASE_DIR / "output.csv"
    output_all_weeks_path: Path = BASE_DIR / "output_all_weeks.csv"
    token_cost_log_path: Path = BASE_DIR / "token_cost_log.json"
    
    # Anomaly detection thresholds
    # According to specifications: flag where cost is rising and looks out of ordinary
    # Sample Row 1: vs_own = +35.5%, vs_peer = +21.0% -> Flagged Yes
    # Sample Row 2: vs_own = +29.5%, vs_peer = +22.5% -> Flagged No (justified)
    # Sample Row 3: vs_own = +9.2%, vs_peer = +23.6%  -> Flagged Yes
    own_history_threshold_pct: float = 20.0
    peer_threshold_pct: float = 20.0
    min_own_rise_pct: float = 0.0  # Must be rising vs own history
    rolling_weeks_window: int = 8
    
    # Pricing for token cost estimation (Gemini 1.5 Flash / GPT-4o-mini rates)
    input_cost_per_million: float = 0.075    # $0.075 / 1M input tokens
    output_cost_per_million: float = 0.300   # $0.300 / 1M output tokens
    
DEFAULT_CONFIG = PipelineConfig()
