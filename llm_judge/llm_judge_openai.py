import os
import json
import requests

# Load OpenAI API key from environment variable
openai_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is loaded; raise an error if not
if openai_key is None:
    raise ValueError("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")

# Load the dataset from a JSON file
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from the dataset.")
    return data

# Function to evaluate each answer
def evaluate_answer(question, ground_truth, generated_answer):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    # Prepare the payload for the API call
    payload = {
        "model": "gpt-4o-mini",  # Specify the model to use for evaluation
        "messages": [
            {
                "role": "user",
                "content": f"""
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
            }
        ],
        "max_tokens": 150  # Limit the number of tokens in the response
    }

    # Make the POST request to the OpenAI API
    print(f"Evaluating answer: {generated_answer[:30]}...")  # Print the start of the answer for context
    response = requests.post(url, headers=headers, json=payload)

    # Check if the response is successful
    if response.status_code == 200:
        # Extract the evaluation from the response
        evaluation = response.json()['choices'][0]['message']['content']
        print("Evaluation received successfully.")
        return evaluation
    else:
        # Print error details and return a message in case of failure
        print(f"Error: {response.status_code}, {response.text}")
        return "Error in evaluation"  # Fallback message

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
    output_file_path = 'evaluation_results_openai.json'

    # Write results to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_file_path}.")

if __name__ == "__main__":
    main()