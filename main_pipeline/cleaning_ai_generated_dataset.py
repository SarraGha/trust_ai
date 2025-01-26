import json
import argparse


def clean_answers(data):
    for entry in data:
        for key in entry['answers']:
            answer = entry['answers'][key]
            # Remove the "\n\nNote:" and everything after it
            if "\n\nNote:" in answer:
                answer = answer.split("\n\nNote:")[0].strip()
            entry['answers'][key] = answer
    return data


def main(input_file):
    try:
        # Load the input data from the specified JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON data: {e}")
        return

    # Clean the data
    cleaned_data = clean_answers(data)

    # Output the cleaned data (you can save it to a file or print it)
    output_file = "ai_generated_dataset_Final_Version_cleaned.json"  # Specify the output file name
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print(f"Cleaned data has been saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean JSON dataset by removing notes.")
    parser.add_argument("--input-file", "-i", required=True, type=str, help="Input JSON dataset file.")

    args = parser.parse_args()

    main(args.input_file)