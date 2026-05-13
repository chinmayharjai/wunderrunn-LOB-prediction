#!/usr/bin/env python
"""
Evaluate all three submissions on the validation set.
"""
import sys
import os

# Add competition package to path for utils
sys.path.insert(0, r"D:\wundernn_fun_challenge\wnn_predictorium_starterpack\competition_package")

from utils import ScorerStepByStep

submissions = {
    "Submission_1": r"D:\wundernn_fun_challenge\wnn_predictorium_starterpack\Submission_1",
    "Submission_2": r"D:\wundernn_fun_challenge\wnn_predictorium_starterpack\Submission_2",
    "Submission_3": r"D:\wundernn_fun_challenge\wnn_predictorium_starterpack\Submission_3",
}

valid_path = r"D:\wundernn_fun_challenge\wnn_predictorium_starterpack\competition_package\datasets\valid.parquet"

results = {}

for name, submission_dir in submissions.items():
    print(f"\n{'='*60}")
    print(f"Evaluating {name}...")
    print(f"{'='*60}")
    
    # Import solution from submission
    sys.path.insert(0, submission_dir)
    
    try:
        # Import the solution module
        import importlib.util
        spec = importlib.util.spec_from_file_location("solution", os.path.join(submission_dir, "solution.py"))
        solution_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution_module)
        
        # Create model
        model = solution_module.PredictionModel()
        
        # Score it
        scorer = ScorerStepByStep(valid_path)
        score = scorer.score(model)
        
        results[name] = score
        
        print(f"\n{name} Results:")
        print(f"  Mean Weighted Pearson Correlation: {score['weighted_pearson']:.6f}")
        for target in scorer.targets:
            print(f"  {target}: {score[target]:.6f}")
            
    except Exception as e:
        print(f"ERROR evaluating {name}: {e}")
        import traceback
        traceback.print_exc()
        results[name] = None
    finally:
        # Remove from path for next iteration
        if submission_dir in sys.path:
            sys.path.remove(submission_dir)

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, score in results.items():
    if score is not None:
        print(f"{name}: {score['weighted_pearson']:.6f}")
    else:
        print(f"{name}: FAILED")

# Find best
valid_scores = {k: v for k, v in results.items() if v is not None}
if valid_scores:
    best_name = max(valid_scores, key=lambda k: valid_scores[k]['weighted_pearson'])
    best_score = valid_scores[best_name]['weighted_pearson']
    print(f"\nBest: {best_name} with {best_score:.6f}")
