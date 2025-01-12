# Imports
import json
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Merge human-generated and AI-generated JSON datasets.')
parser.add_argument('human_file', type=str, help='Path to the human-generated JSON file')
parser.add_argument('ai_file', type=str, help='Path to the AI-generated JSON file')
parser.add_argument('output_file', type=str, help='Path to the output merged JSON file')

# Parse the arguments
args = parser.parse_args()

# Load the human-generated and AI-generated JSON data
with open(args.human_file, 'r') as human_file:
    human_data = json.load(human_file)

with open(args.ai_file, 'r') as ai_file:
    ai_data = json.load(ai_file)

# Prepare the merged data structure
merged_data = {"questions": []}

# Iterate through the questions in the human data
for idx, human_question in enumerate(human_data):
    # Create a new question entry
    question_entry = {
        "id": idx + 1,
        "question": human_question["question"],
        "ground_truth": human_question["ground_truth"],
        "answers": {}
    }

    # Retrieve the corresponding AI answers
    ai_question = ai_data[idx]

    # Populate the answers from both human and AI
    for answer_key in human_question["answers"]:
        question_entry["answers"][answer_key] = {
            "human": human_question["answers"][answer_key],
            "ai": ai_question["answers"][answer_key]
        }

    # Append the question entry to the merged data
    merged_data["questions"].append(question_entry)

# Write the merged data to a new JSON file
with open(args.output_file, 'w') as merged_file:
    json.dump(merged_data, merged_file, indent=3)

print(f"Merging completed. The merged data is saved in '{args.output_file}'.")


