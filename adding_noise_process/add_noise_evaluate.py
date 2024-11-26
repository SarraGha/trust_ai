import json
import random
import requests
import re

# Load Hugging Face token from file
with open('huggingface_token.txt', 'r') as f:
    hf_token = f.read().strip()

# Load the gold dataset from a JSON file
with open('gold_dataset.json', 'r') as f:
    gold_dataset = json.load(f)


# Function to call the Hugging Face Inference API for paraphrasing
def paraphrase_with_huggingface(answer):
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": f"Paraphrase the following sentence: '{answer}'",
        "options": {"use_cache": False}
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        paraphrased_answer = response.json()[0]['generated_text']
        return paraphrased_answer
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return answer  # Return the original answer in case of error


# Function to add noise to answers
def add_noise_to_answers(answer, noise_level):
    if noise_level >= 0.1:
        answer = paraphrase_with_huggingface(answer)

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
altered_datasets = {}

for level in noise_levels:
    altered_dataset = []
    for item in gold_dataset['answers']:
        answer = item['answer']
        altered_answer = add_noise_to_answers(answer, level)
        altered_dataset.append({'answer': altered_answer})
    altered_datasets[level] = altered_dataset

# Display the original dataset
print("Original Dataset:")
for item in gold_dataset['answers']:
    print(item)

# Display altered datasets
for level, altered_data in altered_datasets.items():
    print(f"\nAltered Dataset with Noise Level {level}:")
    for item in altered_data:
        print(item)


# Function to evaluate with Hugging Face model
def evaluate_with_huggingface(data):
    evaluation_prompt = "Evaluate the following answers for clarity and correctness:\n"
    for item in data:
        evaluation_prompt += f"Answer: {item['answer']}\n"

    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": evaluation_prompt,
        "options": {"use_cache": False}
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        evaluation_result = response.json()[0]['generated_text']
        return evaluation_result
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "Evaluation failed."


# Function to extract numerical score from LLM evaluation
def extract_score(evaluation_text):
    # Example: "The clarity is 7/10 and the correctness is 8/10."
    match = re.search(r'(\d+)/10', evaluation_text)
    if match:
        return int(match.group(1))
    return 0  # Default score if no match


# Evaluate each altered dataset with the LLM
llm_evaluations = {}
llm_scores = {}

for level, altered_data in altered_datasets.items():
    evaluation_text = evaluate_with_huggingface(altered_data)
    llm_evaluations[level] = evaluation_text
    llm_scores[level] = extract_score(evaluation_text)  # Extract numerical score
    print(f"LLM Evaluation for Noise Level {level}:\n{evaluation_text}\n")
    print(f"Extracted LLM Score for Noise Level {level}: {llm_scores[level]}")

# Example of manual evaluations with numerical scores
manual_evaluations = {
    0.1: 8,  # Example score out of 10
    0.5: 6,
    1.0: 3
}


# Compare manual evaluations with LLM evaluations
def compare_evaluations(manual_scores, llm_scores):
    evaluation_results = {}

    for level in manual_scores.keys():
        manual_score = manual_scores[level]
        llm_score = llm_scores[level]
        difference = abs(manual_score - llm_score)

        if difference <= 1:
            evaluation_quality = "Well Evaluated"
        elif difference <= 3:
            evaluation_quality = "Moderately Evaluated"
        else:
            evaluation_quality = "Poorly Evaluated"

        evaluation_results[level] = {
            "Manual Score": manual_score,
            "LLM Score": llm_score,
            "Difference": difference,
            "Evaluation Quality": evaluation_quality
        }

    return evaluation_results


# Compare evaluations
comparison_results = compare_evaluations(manual_evaluations, llm_scores)

# Display results
for level, result in comparison_results.items():
    print(f"\nNoise Level {level}:")
    print(f"  Manual Score: {result['Manual Score']}")
    print(f"  LLM Score: {result['LLM Score']}")
    print(f"  Difference: {result['Difference']}")
    print(f"  Evaluation Quality: {result['Evaluation Quality']}\n")