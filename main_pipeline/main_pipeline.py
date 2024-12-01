# Import necessary libraries
import os
import json
import random
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
with open('gold_dataset.json', 'r') as f:
    gold_dataset = json.load(f)

# Function to call the OpenAI API for paraphrasing a given answer
def paraphrase_with_openai(answer):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    # Prepare the payload for the API call
    payload = {
        "model": "gpt-4o-mini",  # Specify the model to use for paraphrasing
        "messages": [
            {"role": "user", "content": f"Paraphrase the following sentence: '{answer}'"}
        ],
        "max_tokens": 60  # Limit the number of tokens in the response
    }

    # Make the POST request to the OpenAI API
    response = requests.post(url, headers=headers, json=payload)

    # Check if the response is successful
    if response.status_code == 200:
        # Extract the paraphrased answer from the response
        paraphrased_answer = response.json()['choices'][0]['message']['content']
        return paraphrased_answer
    else:
        # Print error details and return the original answer in case of failure
        print(f"Error: {response.status_code}, {response.text}")
        return answer  # Fallback to the original answer

# Function to add noise to answers based on a specified noise level
def add_noise_to_answers(answer, noise_level):
    # If noise level is 0.1 or higher, paraphrase the answer
    if noise_level >= 0.1:
        answer = paraphrase_with_openai(answer)

    # If noise level is 0.5 or higher, shuffle sentences and remove random words
    if noise_level >= 0.5:
        sentences = answer.split('. ')  # Split the answer into sentences
        random.shuffle(sentences)  # Shuffle the sentences
        answer = '. '.join(sentences)  # Rejoin shuffled sentences

        words = answer.split()  # Split the answer into words
        num_words_to_remove = max(1, int(len(words) * 0.1))  # Determine number of words to remove (10% of total words)
        for _ in range(num_words_to_remove):
            if words:
                words.remove(random.choice(words))  # Randomly remove words from the answer
        answer = ' '.join(words)  # Rejoin remaining words into a single string

    return answer  # Return the noisy answer

# Specify different noise levels for the evaluation
noise_levels = [0.1, 0.5, 1.0]
final_dataset = []  # Initialize a list to store the final dataset

# Pairing questions with answers based on their index
questions = gold_dataset['questions']
answers = gold_dataset['answers']

# Loop through each noise level and generate noisy answers
for level in noise_levels:
    for i in range(min(len(questions), len(answers))):  # Ensure we don't go out of bounds
        question = questions[i]['question']
        original_answer = answers[i]['answer']
        noisy_answer = add_noise_to_answers(original_answer, level)  # Add noise to the original answer
        # Append the results to the final dataset
        final_dataset.append({
            'user_input': question,
            'reference': original_answer,
            'response': noisy_answer,
            'noise_level': level
        })

# Display the final dataset with noisy answers
print("Final Dataset with Noise:")
for item in final_dataset:
    print(item)

# Function to perform Ragas evaluation on the final dataset
def ragas_evaluation(final_data):
    metrics = [AnswerCorrectness()]  # Specify the metrics to use for evaluation
    dataset = EvaluationDataset.from_dict(final_data)  # Convert the final data into a format suitable for Ragas
    results = evaluate(dataset=dataset, metrics=metrics)  # Evaluate the dataset using the specified metrics
    return results  # Return the evaluation results

# Evaluate the final dataset with Ragas
ragas_results = ragas_evaluation(final_dataset)

# Convert the evaluation results to a pandas DataFrame for easier manipulation
df = ragas_results.to_pandas()

# Add noise levels to the DataFrame for reference
df['noise_level'] = [item['noise_level'] for item in final_dataset]

# Print the Ragas evaluation results for the final dataset
print("\nRagas Evaluation Results for Final Dataset:")
print(df)

# Save the final dataset with noise to a CSV file
final_df = pd.DataFrame(final_dataset)  # Convert the final dataset list to a DataFrame
final_df.to_csv('final_dataset_with_noise.csv', index=False)  # Save to CSV without the index

# Save the Ragas evaluation results to a CSV file
ragas_df = df[['user_input', 'response', 'reference', 'answer_correctness', 'noise_level']]  # Select relevant columns
ragas_df.to_csv('ragas_evaluation_results.csv', index=False)  # Save to CSV without the index

# Example of manual evaluations with numerical scores for comparison
manual_evaluations = {
    0.1: 8,  # Example score for noise level 0.1
    0.5: 6,  # Example score for noise level 0.5
    1.0: 3   # Example score for noise level 1.0
}

# Function to compare manual evaluations with Ragas evaluations
def compare_evaluations(manual_scores, ragas_results):
    evaluation_results = {}  # Initialize a dictionary to store comparison results

    # Group Ragas results by noise level and calculate the mean score for each level
    ragas_scores = ragas_results.groupby('noise_level')['answer_correctness'].mean().to_dict()

    # Loop through each noise level in manual scores for comparison
    for level in manual_scores.keys():
        manual_score = manual_scores[level]  # Get the manual score for the current level
        ragas_score = ragas_scores.get(level, None)  # Get the Ragas score for the current noise level

        if ragas_score is None:
            print(f"Warning: No Ragas score found for noise level {level}")
            continue  # Skip if no score is found

        # Calculate the difference between manual and Ragas scores
        difference = abs(manual_score - ragas_score)
        # Determine the quality of evaluation based on the difference
        evaluation_quality = "Poorly Evaluated" if difference > 5 else "Moderately Evaluated" if difference > 2 else "Well Evaluated"

        # Store the results in the evaluation_results dictionary
        evaluation_results[level] = {
            'manual_score': manual_score,
            'ragas_score': ragas_score,
            'difference': difference,
            'evaluation_quality': evaluation_quality
        }

    return evaluation_results  # Return the comparison results

# Get the evaluation results by comparing manual evaluations with Ragas evaluations
evaluation_results = compare_evaluations(manual_evaluations, df)

# Prepare data for plotting the evaluation results
levels = list(evaluation_results.keys())
manual_scores = [evaluation_results[level]['manual_score'] for level in levels]
ragas_scores = [evaluation_results[level]['ragas_score'] for level in levels]
differences = [evaluation_results[level]['difference'] for level in levels]
evaluation_qualities = [evaluation_results[level]['evaluation_quality'] for level in levels]

# Create a bar plot to visualize the evaluation comparison results
plt.figure(figsize=(10, 6))  # Set the figure size
bar_width = 0.2  # Set the width of the bars
index = range(len(levels))  # Create an index for the x-axis

# Plotting manual scores
plt.bar(index, manual_scores, bar_width, label='Manual Scores', color='b', alpha=0.6)

# Plotting Ragas scores
plt.bar([i + bar_width for i in index], ragas_scores, bar_width, label='Ragas Scores', color='g', alpha=0.6)

# Plotting differences
plt.bar([i + 2 * bar_width for i in index], differences, bar_width, label='Difference', color='r', alpha=0.6)

# Adding labels and title to the plot
plt.xlabel('Noise Level')  # Label for the x-axis
plt.ylabel('Scores')  # Label for the y-axis
plt.title('Evaluation Comparison Results')
# Title of the plot
plt.xticks([i + bar_width for i in index], levels)  # Set x-ticks to noise levels
plt.legend()  # Show legend for the plot

# Adding evaluation quality annotations above the bars
for i, level in enumerate(levels):
    plt.text(i, manual_scores[i] + 0.5, evaluation_qualities[i], ha='center', va='bottom')

plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()  # Display the plot