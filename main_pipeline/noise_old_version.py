# -*- coding: utf-8 -*-
import argparse
import json
import pathlib
import random
import math
import ollama

def generate_response(client, prompt):
    """
    Calls the Ollama API and returns the response text.
    """
    response = client.chat(
        "mistral-nemo:12b-instruct-2407-fp16",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"].strip()

def generate_a0(client, ground_truth):
    prompt = (
        "Rewrite the provided sentence to express the same idea in slightly different words while preserving "
        "full accuracy, completeness, and meaning. Ensure the content remains faithful to the original and includes "
        "all key details.\n\n"
        f"Original:\n\"{ground_truth}\"\n\n"
        "Paraphrased version:"
    )
    return generate_response(client, prompt)

def bracket_and_list_factual_data(client, a0_text):
    prompt = (
        "Take the following text and put each piece of factual data between square brackets. "
        "Then return ONLY a JSON array of these bracketed items. Example output:\n"
        "[\"[Fuel]\", \"[1983]\", \"[Mr. John Colman]\"]\n\n"
        f"Text:\n\"{a0_text}\"\n\n"
        "Output ONLY the JSON array, no extra text."
    )
    response_text = generate_response(client, prompt)

    try:
        factual_data_list = json.loads(response_text)
    except json.JSONDecodeError:
        print("WARNING: Could not parse JSON from model. Response was:", response_text)
        factual_data_list = []

    return factual_data_list

def generate_noisy_version(client, a0_text, items_to_change):
    if not items_to_change:
        return a0_text  # no changes

    bracketed_items_str = ", ".join(items_to_change)
    print(f"Items to change: {bracketed_items_str}")

    prompt = (
        "You must ONLY modify the following bracketed items in the original text, making them slightly incorrect "
        "or omitting them entirely, but always in a realistic and serious manner (no fantasy or humorous content). "
        "All other parts of the text MUST remain unchanged. "
        "Retain the same sentence structure, punctuation, and style wherever possible. "
        "If a bracketed element is a date, change it to a close/near date (e.g., 1809 -> 1810). "
        "If it's a location, substitute it with a plausible alternative in the same era. "
        "If it's a name, role, or numeric value, change it to something similar but not identical. "
        "Remove square brackets [ ] from any changed words.\n\n"
        "Original text:\n"
        f"{a0_text}\n\n"
        "Items to subtly change:\n"
        f"{bracketed_items_str}\n\n"
        "Now produce the final text with only these realistic, minimal changes:"
    )

    noisy_text = generate_response(client, prompt)
    return noisy_text.strip()

def main_pipeline(data):
    client = ollama.Client("http://atlas1api.eurecom.fr:8019")
    results = []

    for entry in data:
        question = entry["question"]
        ground_truth = entry["ground_truth"]

        a0 = generate_a0(client, ground_truth)

        factual_data = bracket_and_list_factual_data(client, a0)

        random.shuffle(factual_data)

        answers = {}
        answers["A0"] = a0
        fractions = [0.25, 0.50, 0.75, 1.0]

        for i, fraction in enumerate(fractions, start=1):
            num_items_to_change = math.ceil(len(factual_data) * fraction)
            items_for_level = factual_data[:num_items_to_change]
            noisy_version = generate_noisy_version(client, a0, items_for_level)
            answers[f"A{i}"] = noisy_version

        result_entry = {
            "question": question,
            "ground_truth": ground_truth,
            "answers": answers
        }
        results.append(result_entry)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--input-file", "-i", required=True, type=pathlib.Path)

    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # data must be a list of {"question": ..., "ground_truth": ...}

    results = main_pipeline(data)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(json.dumps(results, indent=4, ensure_ascii=False))