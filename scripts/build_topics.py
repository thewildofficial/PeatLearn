#!/usr/bin/env python3
import sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "src" / "adaptive_learning" / "topic_model.py"
MODULE_NAME = "src.adaptive_learning.topic_model"

spec = importlib.util.spec_from_file_location(MODULE_NAME, str(MODEL_PATH))
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)  # type: ignore
CorpusTopicModel = module.CorpusTopicModel

def main():
    tm = CorpusTopicModel(
        corpus_glob=str(ROOT / "data" / "processed" / "**" / "*.txt"),
        model_dir=str(ROOT / "data" / "models" / "topics"),
        max_features=50000,
        min_df=2,
        max_df=0.6,
        n_components_svd=300,
    )
    tm.build()
    print("Topics built. Artifacts written to data/models/topics and data/reports/topic_clusters_report.md")

if __name__ == "__main__":
    main()
