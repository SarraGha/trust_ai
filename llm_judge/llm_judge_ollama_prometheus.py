import os
import json
import time
import re  # Import the regular expression module
import ollama

# Load the Ollama API client
client = ollama.Client("http://atlas1api.eurecom.fr:8019")

# Load the dataset from a JSON file
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from the dataset.")
    return data

# Function to generate a response using the Mistral model
def generate_response(prompt):
    response = client.chat(
        "tensortemplar/prometheus2:7b-fp16",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"].strip()

# Function to evaluate each answer
def evaluate_answer(question, ground_truth, generated_answer):
    prompt = f"""
    ### The instruction to evaluate:
    {question}

    ### Response to evaluate:
    {generated_answer}

    ### Reference Answer (Score 5):
    {ground_truth}

    ### Score Rubrics:
    [Are the model's responses factually correct and well-supported by evidence?]
    Score 1: The model's responses are mostly incorrect or based on unfounded information.
    Score 2: The model sometimes provides factually correct responses, but inaccuracies are common.
    Score 3: The model generally provides factually correct information, though some errors occur.
    Score 4: The model often provides factually accurate information with only occasional minor errors.
    Score 5: The model consistently provides responses that are factually correct and well-supported by evidence.

    ### Please provide only a score (1-5) based on the above criteria:
    """

    print(f"Evaluating answer: {generated_answer[:30]}...")  # Print the start of the answer for context
    evaluation = generate_response(prompt)
    print(evaluation)

    # Use regex to find the first numerical value in the evaluation response
    match = re.search(r'\b\d\b', evaluation)  # Match a single digit (1-9)

    if match:
        score = match.group(0)  # Extract the score
    else:
        score = "N/A"  # If no score is found, set to "N/A"

    print(f"Score: {score}")  # Print the score for debugging
    return score

def main():
    # Load the dataset from the specified JSON file
    dataset_file_path = './ai_generated_dataset_llm_judge.json'  # Change this to your actual file path
    dataset = load_dataset(dataset_file_path)

    # Get all uploaded models
    models = client.list()
    available_models = [m["name"] for m in models["models"]]
    my_model = "tensortemplar/prometheus2:7b-fp16"

    if my_model not in available_models:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"Attempting to pull model: {my_model} (Attempt {attempt + 1}/{max_retries})")
                client.pull(my_model)
                print("Model pulled successfully.")
                break  # Exit the loop if successful
            except ollama._types.ResponseError as e:
                print(f"Error pulling model: {e}. Retrying in 5 seconds...")
                time.sleep(5)  # Wait before retrying
        else:
            print("Failed to pull the model after several attempts.")
            return  # Exit the main function if the model cannot be pulled

    results = []

    # Iterate through each question in the dataset
    for entry in dataset:
        question_text = entry['question']
        ground_truth = entry['ground_truth']
        answers = entry['answers']

        print(f"Processing question: {question_text}")  # Print the current question being processed

        # Evaluate each answer
        for answer_key, answer in answers.items():
            evaluation = evaluate_answer(question_text, ground_truth, answer)
            results.append({
                "question": question_text,
                "answer_key": answer_key,
                "evaluation": evaluation
            })

    # Define output file path
    output_file_path = './evaluation_results_ollama_prometheus.json'

    # Write results to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_file_path}.")

if __name__ == "__main__":
    main()