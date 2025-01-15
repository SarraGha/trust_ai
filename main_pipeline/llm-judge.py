# Import necessary libraries
import os
import json
import requests  # Use requests for API calls
from ragas import evaluate, EvaluationDataset
from ragas.metrics import AnswerCorrectness

# Print the current working directory for debugging purposes
print("Current Working Directory:", os.getcwd())

# Load OpenAI API key from environment variable
openai_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is loaded; raise an error if not
if openai_key is None:
    raise ValueError("OpenAI API key not set. Please set the OPENAI_API_KEY environment variable.")

# Load the gold dataset from a JSON file containing questions and answers
with open('evaluation_dataset.json', 'r') as f:
    gold_dataset = json.load(f)


def evaluate_answer(question, ground_truth, human_answer, ai_answer):
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
                Evaluate the following answers to the question: "{question}"

                Ground Truth: {ground_truth}

                Human Answer: {human_answer}

                AI Answer: {ai_answer}

                Please rate the human and AI answers on a scale from 1 to 10 based on their accuracy and relevance to the ground truth. 
                Provide a brief explanation for your ratings.
                """
            }
        ],
        "max_tokens": 150  # Limit the number of tokens in the response
    }

    # Make the POST request to the OpenAI API
    response = requests.post(url, headers=headers, json=payload)

    # Check if the response is successful
    if response.status_code == 200:
        # Extract the evaluation from the response
        evaluation = response.json()['choices'][0]['message']['content']
        return evaluation
    else:
        # Print error details and return a message in case of failure
        print(f"Error: {response.status_code}, {response.text}")
        return "Error in evaluation"  # Fallback message


def main():
    results = []

    for question in gold_dataset['questions']:
        question_id = question['id']
        question_text = question['question']
        ground_truth = question['ground_truth']
        answers = question['answers']

        for answer_key, answer in answers.items():
            human_answer = answer['human']
            ai_answer = answer['ai']

            evaluation = evaluate_answer(question_text, ground_truth, human_answer, ai_answer)
            results.append({
                "question_id": question_id,
                "answer_key": answer_key,
                "evaluation": evaluation
            })

    # Print results
    for result in results:
        print(f"Question ID: {result['question_id']}, Answer Key: {result['answer_key']}")
        print("Evaluation:")
        print(result['evaluation'])
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()