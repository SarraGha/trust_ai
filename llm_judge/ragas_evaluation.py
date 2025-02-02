# -*- coding: utf-8 -*-
"""Ragas Evaluation Script"""

import os
import json
import pandas as pd
from ragas import evaluate, EvaluationDataset
from ragas.metrics import AnswerCorrectness



# Load dataset from a JSON file
with open('ai_generated_dataset_llm_judge.json', 'r') as f:
    dataset_json = json.load(f)

# Convert dataset to the required format for EvaluationDataset
data_samples = []
for item in dataset_json:
    question = item["question"]
    ground_truth = item["ground_truth"]
    answers = item["answers"]

    for key, answer in answers.items():
        data_samples.append({
            'user_input': question,
            'response': answer,
            'reference': ground_truth
        })

# Create EvaluationDataset
dataset = EvaluationDataset.from_dict(data_samples)

# Define the metric to use
metrics = [AnswerCorrectness()]

# Perform evaluation
results = evaluate(dataset=dataset, metrics=metrics)

# Convert results to Pandas DataFrame
df = results.to_pandas()

# Save results to CSV
df.to_csv('ragas_evaluation_results.csv', index=False)

# Print the results
print("\nEvaluation Results:")
print(df)
