import os
import json
import random
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ragas import evaluate, EvaluationDataset
from ragas.metrics import AnswerCorrectness

print("Current Working Directory:", os.getcwd())

# Load OpenAI API key from environment variable
openai_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is loaded
if openai_key is None:
    raise ValueError("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")

# Load the gold dataset from a JSON file
with open('gold_dataset.json', 'r') as f:
    gold_dataset = json.load(f)

# Function to call the OpenAI API for paraphrasing
def paraphrase_with_openai(answer):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",  # You can change this to a different model if needed
        "messages": [
            {"role": "user", "content": f"Paraphrase the following sentence: '{answer}'"}
        ],
        "max_tokens": 60
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        paraphrased_answer = response.json()['choices'][0]['message']['content']
        return paraphrased_answer
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return answer  # Return the original answer in case of error

# Function to add noise to answers
def add_noise_to_answers(answer, noise_level):
    if noise_level >= 0.1:
        answer = paraphrase_with_openai(answer)

    if noise_level >= 0.5:
        sentences = answer.split('. ')
        random.shuffle(sentences)
        answer = '. '.join(sentences)

        words = answer.split()
        num_words_to_remove = max(1, int(len(words) * 0.1))
        for _ in range(num_words_to_remove):
            if words:
                words.remove(random.choice(words))
        answer = ' '.join(words)

    return answer

# Add noise at three different levels
noise_levels = [0.1, 0.5, 1.0]
final_dataset = []

# Pairing questions with answers based on their index
questions = gold_dataset['questions']
answers = gold_dataset['answers']

for level in noise_levels:
    for i in range(min(len(questions), len(answers))):  # Ensure we don't go out of bounds
        question = questions[i]['question']
        original_answer = answers[i]['answer']
        noisy_answer = add_noise_to_answers(original_answer, level)
        final_dataset.append({
            'user_input': question,
            'reference': original_answer,
            'response': noisy_answer,
            'noise_level': level
        })

# Display the final dataset
print("Final Dataset with Noise:")
for item in final_dataset:
    print(item)

# Ragas Evaluation
def ragas_evaluation(final_data):
    metrics = [AnswerCorrectness()]
    dataset = EvaluationDataset.from_dict(final_data)
    results = evaluate(dataset=dataset, metrics=metrics)
    return results

# Evaluate the final dataset with Ragas
ragas_results = ragas_evaluation(final_dataset)

# Convert results to pandas DataFrame
df = ragas_results.to_pandas()

# Add noise levels to the DataFrame
df['noise_level'] = [item['noise_level'] for item in final_dataset]

print("\nRagas Evaluation Results for Final Dataset:")
print(df)

# Example of manual evaluations with numerical scores
manual_evaluations = {
    0.1: 8,  # Example score out of 10
    0.5: 6,
    1.0: 3
}

# Function to compare manual evaluations with Ragas evaluations
def compare_evaluations(manual_scores, ragas_results):
    evaluation_results = {}

    # Group results by noise level
    ragas_scores = ragas_results.groupby('noise_level')['answer_correctness'].mean().to_dict()

    for level in manual_scores.keys():
        manual_score = manual_scores[level]
        ragas_score = ragas_scores.get(level, None)  # Get the Ragas score for the current noise level

        if ragas_score is None:
            print(f"Warning: No Ragas score found for noise level {level}")
            continue  # Skip if no score is found

        difference = abs(manual_score - ragas_score)
        evaluation_quality = "Poorly Evaluated" if difference > 5 else "Moderately Evaluated" if difference > 2 else "Well Evaluated"

        evaluation_results[level] = {
            'manual_score': manual_score,
            'ragas_score': ragas_score,
            'difference': difference,
            'evaluation_quality': evaluation_quality
        }

    return evaluation_results

# Get the evaluation results
evaluation_results = compare_evaluations(manual_evaluations, df)

# Prepare data for plotting
levels = list(evaluation_results.keys())
manual_scores = [evaluation_results[level]['manual_score'] for level in levels]
ragas_scores = [evaluation_results[level]['ragas_score'] for level in levels]
differences = [evaluation_results[level]['difference'] for level in levels]
evaluation_qualities = [evaluation_results[level]['evaluation_quality'] for level in levels]

# Create a plot
plt.figure(figsize=(10, 6))
bar_width = 0.2
index = range(len(levels))

# Plotting manual scores
plt.bar(index, manual_scores, bar_width, label='Manual Scores', color='b', alpha=0.6)

# Plotting Ragas scores
plt.bar([i + bar_width for i in index], ragas_scores, bar_width, label='Ragas Scores', color='g', alpha=0.6)

# Plotting differences
plt.bar([i + 2 * bar_width for i in index], differences, bar_width, label='Difference', color='r', alpha=0.6)

# Adding labels and title
plt.xlabel('Noise Level')
plt.ylabel('Scores')
plt.title('Evaluation Comparison Results')
plt.xticks([i + bar_width for i in index], levels)
plt.legend()

# Adding evaluation quality annotations
for i, level in enumerate(levels):
    plt.text(i, manual_scores[i] + 0.5, evaluation_qualities[i], ha='center', va='bottom')

plt.tight_layout()
plt.show()