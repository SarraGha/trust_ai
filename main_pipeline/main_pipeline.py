import os
import json
import random
import requests
import pandas as pd
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

# Convert results to pandas DataFrame and display
df = ragas_results.to_pandas()
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
    # Check if 'answer_correctness' is in the DataFrame
    if 'answer_correctness' in ragas_results.columns:
        for level in manual_scores.keys():
            manual_score = manual_scores[level]
            ragas_score = ragas_results.loc[ragas_results[
                                                'noise_level'] == level, 'answer_correctness'].mean()  # Adjust based on your DataFrame structure
            difference = abs(manual_score - ragas_score)

            if difference <= 1:
                evaluation_quality = "Well Evaluated"
            elif difference <= 3:
                evaluation_quality = "Moderately Evaluated"
            else:
                evaluation_quality = "Poorly Evaluated"

            evaluation_results