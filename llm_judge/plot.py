import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Function to load scores from a JSON file
def load_scores(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    scores = {f'A{i}': [] for i in range(5)}  # Initialize scores for A0 to A4
    for entry in data:
        try:
            score_str = entry['evaluation']
            if '/' in score_str:
                score_str = score_str.split('/')[0]  # Take the numerator
            score = float(score_str)  # Convert to float
            scores[entry['answer_key']].append(score)
        except (ValueError, IndexError):
            scores[entry['answer_key']].append(None)

    return scores

# Function to calculate the mean of a list, ignoring None values
def safe_mean(values):
    """
    Calculate the mean of a list, ignoring None values.
    Returns None if all values are None.
    """
    filtered_values = [v for v in values if v is not None]
    return np.mean(filtered_values) if filtered_values else None

# Function to evaluate a single metric and plot results
def evaluate_single_metric(real_noise, predicted_values, metric_name):
    """
    Calculate Pearson Correlation for a single metric and create plots.
    Parameters:
    real_noise (list or numpy array): The ground truth noise levels (e.g., A0 to A4).
    predicted_values (list or numpy array): Predicted noise levels from the metric.
    metric_name (str): Name of the metric.
    Returns:
    None
    """
    if len(real_noise) != len(predicted_values):
        raise ValueError(f"Length of predictions for {metric_name} does not match real noise.")

    # Compute Pearson Correlation
    corr, _ = pearsonr(real_noise, predicted_values)

    # Plot scatter for the current metric
    plt.figure(figsize=(8, 6))
    plt.scatter(real_noise, predicted_values, alpha=0.7, label=f"{metric_name} (r={corr:.2f})")
    plt.plot(np.unique(real_noise),
             np.poly1d(np.polyfit(real_noise, predicted_values, 1))(np.unique(real_noise)),
             color='red', linestyle='--', label="Best Fit Line")
    plt.title(f"Performance of {metric_name}", fontsize=14)
    plt.xlabel("Real Noise Levels", fontsize=12)
    plt.ylabel(f"Predicted Noise by {metric_name}", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

    print(f"Pearson Correlation for {metric_name}: {corr:.2f}")

# Load scores from JSON files
scores_mistral = load_scores('evaluation_results_ollama_mistral.json')
scores_llama3 = load_scores('evaluation_results_ollama_llama3.json')
scores_openai = load_scores('evaluation_results_openai.json')
scores_prometheus = load_scores('evaluation_results_ollama_prometheus.json')

# Prepare noise levels and align scores
noise_levels = ['A0', 'A1', 'A2', 'A3', 'A4']
real_noise = [safe_mean(scores_mistral[level]) for level in noise_levels]
llama3_scores = [safe_mean(scores_llama3[level]) for level in noise_levels]
openai_scores = [safe_mean(scores_openai[level]) for level in noise_levels]
prometheus_scores = [safe_mean(scores_prometheus[level]) for level in noise_levels]

# Filter out None values to ensure alignment for plotting and correlation
def filter_valid_scores(real, pred):
    return zip(*[(r, p) for r, p in zip(real, pred) if r is not None and p is not None])

# Evaluate and plot each metric
for metric_scores, metric_name in zip([llama3_scores, openai_scores, prometheus_scores, real_noise],
                                      ["Llama3", "OpenAI", "Prometheus", "Mistral"]):
    valid_real, valid_pred = filter_valid_scores(real_noise, metric_scores)
    evaluate_single_metric(valid_real, valid_pred, metric_name)