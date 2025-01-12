import json
import re
import pathlib
import argparse

def parse_text_file(file_path: pathlib.Path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into sections based on the delimiter
    sections = content.strip().split('----')
    questions = []

    for section in sections:
        # Extract question and ground truth
        question_match = re.search(r'question:\s*(.*)', section)
        ground_truth_match = re.search(r'ground_truth:\s*(.*)', section)

        if question_match and ground_truth_match:
            question = question_match.group(1).strip()
            ground_truth = ground_truth_match.group(1).strip()

            # Extract answers (a0, a1, a2, a3, a4)
            answers = {}
            for i in range(5):
                answer_match = re.search(r'a' + str(i) + r':\s*(.*)', section)
                if answer_match:
                    answers[f'A{i}'] = {
                        "human": answer_match.group(1).strip(),
                        "ai": None  # Placeholder for AI answers
                    }

            questions.append({
                "id": len(questions) + 1,
                "question": question,
                "ground_truth": ground_truth,
                "answers": answers
            })

    return questions

def merge_ai_data(questions, ai_data):
    for question in questions:
        question_text = question["question"]
        for i in range(5):
            if i < len(ai_data):
                question["answers"][f'A{i}']["ai"] = ai_data[i]["answer"]
    return questions

def save_to_json(data, output_file: pathlib.Path):
    with open(output_file, 'w') as f:
        json.dump({"questions": data}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge human-generated and AI-generated datasets.")
    parser.add_argument("--input-text", "-it", required=True, type=pathlib.Path, help="Input text file containing human-generated data.")
    parser.add_argument("--input-json", "-ij", required=True, type=pathlib.Path, help="Input JSON file containing AI-generated data.")
    parser.add_argument("--output-file", "-o", required=True, type=pathlib.Path, help="Output JSON file for merged data.")

    args = parser.parse_args()

    # Parse the human-generated text file
    questions_data = parse_text_file(args.input_text)

    # Load the AI-generated data from the JSON file
    with open(args.input_json, 'r') as json_file:
        ai_data = json.load(json_file)

    # Merge AI data into the questions
    merged_data = merge_ai_data(questions_data, ai_data)

    # Save the merged data to the output file
    save_to_json(merged_data, args.output_file)

    print(f"Data has been successfully written to {args.output_file}")