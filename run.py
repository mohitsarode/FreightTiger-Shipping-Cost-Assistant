import argparse
import sys
from pathlib import Path
from src.config import DEFAULT_CONFIG
from src.pipeline import ShippingCostPipeline
from eval.eval_harness import EvaluationHarness
from scripts.verify_reproducibility import verify_3_runs
from src.assistant import FreightCostAssistant

def main():
    parser = argparse.ArgumentParser(
        description="FreightTiger Smart Assistant That Watches Shipping Costs",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run end-to-end processing pipeline and generate output.csv"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation harness scoring against labeled ground truth and trap notes"
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Run reproducibility verification across 3 untouched runs (SHA-256 diff)"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch interactive plain-English Q&A session with the assistant"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the interactive web dashboard (FastAPI)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run pipeline, eval harness, and reproducibility verification sequentially"
    )

    args = parser.parse_args()

    # Default to running pipeline if no args provided
    if len(sys.argv) == 1 or args.pipeline:
        print("[1/1] Running FreightTiger Shipping Cost Pipeline...")
        pipeline = ShippingCostPipeline(DEFAULT_CONFIG)
        summary = pipeline.run()
        print("\nPipeline Execution Successful!")
        print(f"Total Shipments  : {summary['total_shipments_processed']}")
        print(f"Total Route-Weeks: {summary['total_route_weeks']}")
        print(f"Evaluated Spikes : {summary['candidate_anomalies_evaluated']}")
        print(f"Output File      : {summary['output_files']['primary_anomalies_csv']}")
        print(f"Token Log        : {summary['output_files']['token_log']}")
        return

    if args.eval:
        print("Running Evaluation Harness...")
        harness = EvaluationHarness()
        harness.run_eval()
        return

    if args.reproduce:
        print("Running Reproducibility Audit...")
        verify_3_runs()
        return

    if args.all:
        print("=" * 65)
        print("EXECUTING COMPLETE FREIGHTTIGER SUBMISSION SUITE")
        print("=" * 65)
        print("\n--- STEP 1: PIPELINE RUN ---")
        pipeline = ShippingCostPipeline(DEFAULT_CONFIG)
        pipeline.run()
        
        print("\n--- STEP 2: EVALUATION HARNESS ---")
        harness = EvaluationHarness()
        harness.run_eval()
        
        print("\n--- STEP 3: REPRODUCIBILITY AUDIT ---")
        verify_3_runs()
        return

    if args.chat:
        print("=" * 65)
        print("FREIGHTTIGER SMART ASSISTANT INTERACTIVE CONSOLE")
        print("Type your question in plain English (or 'exit' to quit).")
        print("=" * 65)
        assistant = FreightCostAssistant()
        while True:
            try:
                user_q = input("\nYou: ").strip()
                if not user_q or user_q.lower() in ["exit", "quit", "q"]:
                    print("Exiting assistant. Goodbye!")
                    break
                ans = assistant.answer_question(user_q)
                print(f"\nAssistant:\n{ans}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break
        return

    if args.web:
        import uvicorn
        print("Starting FreightTiger Web Dashboard at http://localhost:8000 ...")
        uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)
        return

if __name__ == "__main__":
    main()
