# -*- coding: utf-8 -*-
"""BLEU and ROUGE Evaluation Script"""

# Import necessary libraries
import json
import pandas as pd
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Load spaCy model for tokenization
nlp = spacy.load("en_core_web_sm")


# Define BLEU and ROUGE scoring functions
def compute_bleu_score_spacy(reference, hypothesis):
    """
    Computes the BLEU score between a reference and a hypothesis sentence using spaCy for tokenization.
    """
    reference_tokens = [token.text for token in nlp(reference)]
    hypothesis_tokens = [token.text for token in nlp(hypothesis)]
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)
    return bleu


def compute_rouge_scores(reference, hypothesis):
    """
    Computes the ROUGE-1, ROUGE-2, and ROUGE-L F1 scores using rouge-score.
    Returns a dict with three ROUGE metrics.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


# Load the dataset from a JSON file
with open('ai_generated_dataset_llm_judge.json', 'r') as f:
    dataset = json.load(f)

# Initialize a list to store evaluation results
results = []

# Loop through each question in the dataset
for item in dataset:
    ground_truth = item["ground_truth"]
    question = item["question"]
    answers = item["answers"]

    # Evaluate each answer (A0-A4)
    for key, answer in answers.items():
        # Compute BLEU score
        bleu_score = compute_bleu_score_spacy(ground_truth, answer)

        # Compute ROUGE scores
        rouge_scores = compute_rouge_scores(ground_truth, answer)

        # Append results to the list
        results.append({
            'question': question,
            'answer_key': key,
            'bleu_score': bleu_score,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL']
        })

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('bleu_rouge_evaluation_results.csv', index=False)

# Print the results
print(results_df)