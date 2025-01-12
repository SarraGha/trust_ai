import argparse
import json
import pathlib


def extract_questions_and_ground_truths(input_file: pathlib.Path, output_file: pathlib.Path):
    with open(input_file, 'r') as f:
        content = f.read()

    # Split the content into sections based on the 'question:' keyword
    sections = content.split('question:')

    extracted_data = []

    for section in sections[1:]:  # Skip the first split part as it will be empty
        lines = section.strip().split('\n')
        question = lines[0].strip()  # The first line is the question
        ground_truth = None

        # Find the ground_truth line
        for line in lines:
            if line.startswith('ground_truth:'):
                ground_truth = line[len('ground_truth:'):].strip()
                break

        if question and ground_truth:
            extracted_data.append({
                "question": question,
                "ground_truth": ground_truth
            })

    # Write the extracted data to the output file
    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract questions and ground truths from a structured dataset.")
    parser.add_argument("--input-file", "-i", required=True, type=pathlib.Path,
                        help="Input file containing the dataset.")
    parser.add_argument("--output-file", "-o", required=True, type=pathlib.Path,
                        help="Output file for the extracted questions and ground truths.")

    args = parser.parse_args()

    extract_questions_and_ground_truths(args.input_file, args.output_file)