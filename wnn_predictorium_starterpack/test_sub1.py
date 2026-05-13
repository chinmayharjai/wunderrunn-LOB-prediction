import sys
import os

# Test Submission 1
sys.path.insert(0, r"D:\wundernn_fun_challenge\wnn_predictorium_starterpack\competition_package")
sys.path.insert(0, r"D:\wundernn_fun_challenge\wnn_predictorium_starterpack\Submission_1")

from utils import ScorerStepByStep

valid_path = r"D:\wundernn_fun_challenge\wnn_predictorium_starterpack\competition_package\datasets\valid.parquet"

from solution import PredictionModel

model = PredictionModel()
scorer = ScorerStepByStep(valid_path)
score = scorer.score(model)

print(f"Submission_1: {score['weighted_pearson']:.6f}")
for target in scorer.targets:
    print(f"  {target}: {score[target]:.6f}")
