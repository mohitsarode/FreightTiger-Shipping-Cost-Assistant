import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_CONFIG
from src.pipeline import ShippingCostPipeline

if __name__ == "__main__":
    pipeline = ShippingCostPipeline(DEFAULT_CONFIG)
    summary = pipeline.run()
    print("Pipeline finished successfully.")
