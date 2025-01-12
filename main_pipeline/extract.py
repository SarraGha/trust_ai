import argparse
import json
import pathlib

def extract_questions_and_ground_truths(input_file: pathlib.Path, output_file: pathlib.Path):
    with open(input_file, 'r') as f:
        data = json.load(f)

    extracted_data = []

    for item in data:
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        if question and ground_truth:
            extracted_data.append({
                "question": question,
                "ground_truth": ground_truth
            })

    # Write the extracted data to the output file
    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract questions and ground truths from a structured JSON dataset.")
    parser.add_argument("--input-file", "-i", required=True, type=pathlib.Path,
                        help="Input JSON file containing the dataset.")
    parser.add_argument("--output-file", "-o", required=True, type=pathlib.Path,
                        help="Output JSON file for the extracted questions and ground truths.")

    args = parser.parse_args()

    extract_questions_and_ground_truths(args.input_file, args.output_file)