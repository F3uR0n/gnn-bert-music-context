# evaluate.py
# Execution wrapper for evaluation pipeline

import os
import json

def main():
    print("========================================")
    print("EVALUATION PIPELINE")
    print("========================================")
    metrics_path = "../results/metrics.json"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            print("Successfully loaded evaluation metrics:")
            print(json.dumps(metrics, indent=2))
    else:
        print("Metrics file not found. Please run the training notebooks first to generate results/metrics.json.")
        
    print("\nFor qualitative retrieval evaluation and t-SNE visualizations, please refer to task_3 and task_4 notebooks.")

if __name__ == "__main__":
    main()