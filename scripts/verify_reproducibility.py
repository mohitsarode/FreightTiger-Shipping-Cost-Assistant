import hashlib
import sys
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PipelineConfig, DEFAULT_CONFIG
from src.pipeline import ShippingCostPipeline

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_3_runs():
    print("=" * 65)
    print("REPRODUCIBILITY VERIFICATION ACROSS 3 UNTOUCHED RUNS")
    print("=" * 65)
    
    run_paths = []
    hashes = []
    dfs = []
    
    for run_idx in [1, 2, 3]:
        out_p = ROOT_DIR / f"output_run_{run_idx}.csv"
        cfg = PipelineConfig(
            shipment_records_path=DEFAULT_CONFIG.shipment_records_path,
            context_notes_path=DEFAULT_CONFIG.context_notes_path,
            sample_output_path=DEFAULT_CONFIG.sample_output_path,
            output_path=out_p,
            output_all_weeks_path=ROOT_DIR / f"output_all_run_{run_idx}.csv",
            token_cost_log_path=ROOT_DIR / f"token_cost_run_{run_idx}.json"
        )
        print(f"\nExecuting Pipeline Run #{run_idx}...")
        pipeline = ShippingCostPipeline(cfg)
        pipeline.run()
        
        sha = file_hash(out_p)
        df = pd.read_csv(out_p, dtype=str)
        run_paths.append(out_p)
        hashes.append(sha)
        dfs.append(df)
        print(f"Run #{run_idx} complete -> SHA-256: {sha}")

    # Check Hash equality
    all_hashes_equal = (hashes[0] == hashes[1] == hashes[2])
    
    # Check DataFrame equivalence
    df1, df2, df3 = dfs[0], dfs[1], dfs[2]
    num_rows_equal = (len(df1) == len(df2) == len(df3))
    
    # Compare critical columns: numbers, flags, and cited notes
    crit_cols = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]
    cols_1_2_equal = df1[crit_cols].equals(df2[crit_cols])
    cols_2_3_equal = df2[crit_cols].equals(df3[crit_cols])
    explanations_equal = df1["reason"].equals(df2["reason"]) and df2["reason"].equals(df3["reason"])

    print("\n" + "=" * 65)
    print("REPRODUCIBILITY AUDIT RESULTS")
    print(f"Run 1 SHA-256: {hashes[0]}")
    print(f"Run 2 SHA-256: {hashes[1]}")
    print(f"Run 3 SHA-256: {hashes[2]}")
    print(f"Byte-for-Byte Exact Hash Match Across 3 Runs : {'YES' if all_hashes_equal else 'NO'}")
    print(f"Row Count Consistency ({len(df1)} rows)             : {'YES' if num_rows_equal else 'NO'}")
    print(f"Critical Columns Identical Across 3 Runs      : {'YES' if (cols_1_2_equal and cols_2_3_equal) else 'NO'}")
    print(f"Reasoning Prose Identical Across 3 Runs       : {'YES' if explanations_equal else 'NO'}")
    print("=" * 65)

    # Clean up temporary run files
    for p in run_paths:
        if p.exists():
            p.unlink()
    for idx in [1, 2, 3]:
        all_p = ROOT_DIR / f"output_all_run_{idx}.csv"
        tok_p = ROOT_DIR / f"token_cost_run_{idx}.json"
        if all_p.exists(): all_p.unlink()
        if tok_p.exists(): tok_p.unlink()

    if all_hashes_equal and cols_1_2_equal and cols_2_3_equal:
        print("\nALL 3 RUNS MATCH 100% BIT-FOR-BIT! PERFECT REPRODUCIBILITY CONFIRMED.")
        return True
    else:
        print("\nREPRODUCIBILITY CHECK FAILED.")
        return False

if __name__ == "__main__":
    success = verify_3_runs()
    sys.exit(0 if success else 1)
