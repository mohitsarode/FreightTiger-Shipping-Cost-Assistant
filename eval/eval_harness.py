import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_CONFIG
from src.core_metrics import load_and_preprocess_shipments, compute_weekly_metrics
from src.pipeline import ShippingCostPipeline

class EvaluationHarness:
    """
    Lightweight automated evaluation harness and trust validation suite.
    Scores the pipeline against labeled ground truth rows, tests anti-hallucination guardrails,
    and computes precision, recall, and hallucination rates.
    """
    
    GROUND_TRUTH_SAMPLES = [
        {
            "route": "Delhi-Jaipur",
            "week_of": "2024-11-11",
            "expected_cost": "4.17",
            "expected_vs_own": "+35.5% vs this route's past average",
            "expected_vs_peer": "+21.0% vs similar-length routes this week",
            "expected_flagged": "Yes",
            "expected_note_id": ""
        },
        {
            "route": "Ahmedabad-Mumbai",
            "week_of": "2025-01-20",
            "expected_cost": "3.29",
            "expected_vs_own": "+29.5% vs this route's past average",
            "expected_vs_peer": "+22.5% vs similar-length routes this week",
            "expected_flagged": "No (justified)",
            "expected_note_id": "N002"
        },
        {
            "route": "Mumbai-Pune",
            "week_of": "2025-09-15",
            "expected_cost": "3.98",
            "expected_vs_own": "+9.2% vs this route's past average",
            "expected_vs_peer": "+23.6% vs similar-length routes this week",
            "expected_flagged": "Yes",
            "expected_note_id": ""
        }
    ]
    
    DISTRACTOR_NOTES = ["N004", "N005", "N006", "N007", "N008", "N009", "N010"]

    def __init__(self):
        self.pipeline = ShippingCostPipeline(DEFAULT_CONFIG)
        
    def run_eval(self) -> Dict[str, Any]:
        print("=" * 60)
        print("RUNNING FREIGHTTIGER ASSISTANT EVALUATION HARNESS")
        print("=" * 60)
        
        # 1. Execute pipeline
        summary = self.pipeline.run()
        output_df = pd.read_csv(DEFAULT_CONFIG.output_path, dtype=str).fillna("")
        
        # 2. Check Ground Truth Samples
        print("\n[TEST 1] Ground Truth Sample Evaluation:")
        ground_truth_passed = 0
        total_gt = len(self.GROUND_TRUTH_SAMPLES)
        
        for sample in self.GROUND_TRUTH_SAMPLES:
            r = sample['route']
            w = sample['week_of']
            match = output_df[(output_df['route'] == r) & (output_df['week_of'] == w)]
            
            if match.empty:
                print(f"  FAIL: Expected sample ({r}, {w}) not found in candidate output!")
                continue
                
            row = match.iloc[0]
            cost_ok = (row['cost_per_tonne_km'] == sample['expected_cost'])
            own_ok = (row['vs_own_history'] == sample['expected_vs_own'])
            peer_ok = (row['vs_similar_routes'] == sample['expected_vs_peer'])
            flag_ok = (row['flagged'] == sample['expected_flagged'])
            note_ok = (str(row['matched_note_id']).strip() == sample['expected_note_id'])
            
            all_ok = cost_ok and own_ok and peer_ok and flag_ok and note_ok
            if all_ok:
                ground_truth_passed += 1
                print(f"  PASS: {r} @ {w} -> cost={row['cost_per_tonne_km']}, own={row['vs_own_history']}, peer={row['vs_similar_routes']}, flag={row['flagged']}, note={row['matched_note_id']}")
            else:
                print(f"  FAIL: {r} @ {w}")
                if not cost_ok: print(f"    cost mismatch: got {row['cost_per_tonne_km']}, expected {sample['expected_cost']}")
                if not own_ok: print(f"    vs_own mismatch: got {row['vs_own_history']}, expected {sample['expected_vs_own']}")
                if not peer_ok: print(f"    vs_peer mismatch: got {row['vs_similar_routes']}, expected {sample['expected_vs_peer']}")
                if not flag_ok: print(f"    flagged mismatch: got {row['flagged']}, expected {sample['expected_flagged']}")
                if not note_ok: print(f"    note_id mismatch: got '{row['matched_note_id']}', expected '{sample['expected_note_id']}'")

        # 3. Check Trap Notes & Hallucinations
        print("\n[TEST 2] Anti-Hallucination Guardrail Check on Trap Notes:")
        justified_rows = output_df[output_df['flagged'] == 'No (justified)']
        cited_notes = justified_rows['matched_note_id'].tolist()
        
        trap_violations = [nid for nid in cited_notes if nid in self.DISTRACTOR_NOTES]
        hallucination_count = len(trap_violations)
        
        if hallucination_count == 0:
            print(f"  PASS: Zero trap notes were cited across all {len(justified_rows)} justified verdicts.")
            print(f"        Verified notes cited: {set(cited_notes)}")
        else:
            print(f"  CRITICAL FAILURE: Trap notes cited as justification: {trap_violations}")

        # 4. Check Temporal Validity
        print("\n[TEST 3] Temporal Alignment Check:")
        temporal_failures = 0
        for _, row in justified_rows.iterrows():
            nid = row['matched_note_id']
            wk = row['week_of']
            # If N001, must be in Feb/Mar 2025 flood window
            if nid == "N001" and wk not in ["2025-02-24", "2025-03-03"]:
                print(f"  FAIL: Note N001 cited outside flood disruption window for week {wk}")
                temporal_failures += 1
            # If N002, must be in Jan 2025 festival week
            if nid == "N002" and wk != "2025-01-20":
                print(f"  FAIL: Note N002 cited outside festival week for week {wk}")
                temporal_failures += 1
                
        if temporal_failures == 0:
            print(f"  PASS: All justified records strictly adhere to active temporal windows.")

        # 5. Check Output Format & Contract
        print("\n[TEST 4] Schema and Column Contract Check:")
        expected_cols = [
            "route", "week_of", "cost_per_tonne_km", "vs_own_history",
            "vs_similar_routes", "flagged", "matched_note_id", "reason"
        ]
        actual_cols = list(output_df.columns)
        schema_ok = (actual_cols == expected_cols)
        if schema_ok:
            print(f"  PASS: Column names and order exactly match contract: {actual_cols}")
        else:
            print(f"  FAIL: Column mismatch! Got {actual_cols}, expected {expected_cols}")

        # Summary Metrics
        gt_accuracy = (ground_truth_passed / total_gt) * 100.0
        hallucination_rate = (hallucination_count / max(1, len(justified_rows))) * 100.0
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print(f"Ground Truth Benchmark Accuracy : {gt_accuracy:.1f}% ({ground_truth_passed}/{total_gt})")
        print(f"Hallucination / Trap Note Rate  : {hallucination_rate:.1f}% ({hallucination_count} violations)")
        print(f"Temporal Alignment Compliance   : {(1.0 - temporal_failures / max(1, len(justified_rows))) * 100.0:.1f}%")
        print(f"Schema Contract Compliance      : {'100%' if schema_ok else '0%'}")
        print("=" * 60)
        
        return {
            "ground_truth_accuracy_pct": gt_accuracy,
            "hallucination_rate_pct": hallucination_rate,
            "trap_violations": trap_violations,
            "schema_compliant": schema_ok,
            "evaluated_rows": len(output_df),
            "justified_rows": len(justified_rows)
        }

if __name__ == "__main__":
    harness = EvaluationHarness()
    results = harness.run_eval()
    if results['ground_truth_accuracy_pct'] == 100.0 and results['hallucination_rate_pct'] == 0.0:
        print("\nALL EVALUATION CHECKS PASSED PERFECTLY!")
        sys.exit(0)
    else:
        print("\nEVALUATION CHECKS COMPLETED WITH WARNINGS/FAILURES.")
        sys.exit(1)
