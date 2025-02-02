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


if __name__=="__main__":
    # Load scores from JSON files
    scores_mistral = load_scores('evaluation_results_ollama_mistral.json')
    scores_llama3 = load_scores('evaluation_results_ollama_llama3.json')
    scores_openai = load_scores('evaluation_results_openai.json')
    scores_prometheus = load_scores('evaluation_results_ollama_prometheus.json')

    # Prepare noise levels and align scores
    noise_levels = ['A0', 'A1', 'A2', 'A3', 'A4']
    d = {"A0": 5, "A1": 4, "A2": 3, "A3": 2, "A4": 1}

    # Mistral Nemo pearson evaluation
    output_mistral = []
    for l in noise_levels:
        for s in scores_mistral[l]:
            if s is not None:
                output_mistral.append((s, d[l]))

    output_mistral = np.array(output_mistral)
    output_mistral += np.random.normal(loc=0, scale=0.15, size=output_mistral.shape)
    evaluate_single_metric([o[1] for o in output_mistral], [o[0] for o in output_mistral], "Mistral-Nemo")

    # Llama3.3 pearson evaluation
    output_llama = []
    for l in noise_levels:
        for s in scores_llama3[l]:
            if s is not None:
                output_llama.append((s, d[l]))

    output_llama = np.array(output_llama)
    output_llama += np.random.normal(loc=0, scale=0.15, size=output_llama.shape)
    evaluate_single_metric([o[1] for o in output_llama], [o[0] for o in output_llama], "Llama 3.3")

    # GPT4o pearson evaluation
    output_gpt4o = []
    for l in noise_levels:
        for s in scores_openai[l]:
            if s is not None:
                output_gpt4o.append((s, d[l]))

    output_gpt4o = np.array(output_gpt4o)
    output_gpt4o += np.random.normal(loc=0, scale=0.15, size=output_gpt4o.shape)
    evaluate_single_metric([o[1] for o in output_gpt4o], [o[0] for o in output_gpt4o], "GPT4o")

    # Prometheus pearson evaluation
    output_prometheus = []
    for l in noise_levels:
        for s in scores_prometheus[l]:
            if s is not None:
                output_prometheus.append((s, d[l]))

    output_prometheus = np.array(output_prometheus)
    output_prometheus += np.random.normal(loc=0, scale=0.15, size=output_prometheus.shape)
    evaluate_single_metric([o[1] for o in output_prometheus], [o[0] for o in output_prometheus], "Prometheus")
