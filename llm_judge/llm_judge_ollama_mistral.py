import os
import json
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
        "mistral-nemo:12b-instruct-2407-fp16",
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

    ### Please provide a score from 1 to 5 based on the above criteria:
    """

    print(f"Evaluating answer: {generated_answer[:30]}...")  # Print the start of the answer for context
    evaluation = generate_response(prompt)

    # Extract only the score from the evaluation response
    score = evaluation.split()[1][0]  # Assuming the score is the first word in the response
    print(score)
    return score

def main():
    # Load the dataset from the specified JSON file
    dataset_file_path = './ai_generated_dataset_llm_judge.json'  # Change this to your actual file path
    dataset = load_dataset(dataset_file_path)

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
    output_file_path = './evaluation_results_ollama_mistral.json'

    # Write results to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_file_path}.")

if __name__ == "__main__":
    main()