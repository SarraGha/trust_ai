# -*- coding: utf-8 -*-
import argparse
import json
import pathlib
import random
import math
import re
import ollama

def generate_response_ollama(client, prompt, instructions):
    """
    Calls the Ollama API and returns the response text.
    """
    response = client.chat(
        "llama3.3:70b",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"].strip().replace("\"", "")

def generate_a0(client, ground_truth):
    prompt = (
        "Rewrite the provided sentence to express the same idea in slightly different words while preserving "
        "full accuracy, completeness, and meaning. Ensure the content remains faithful to the original and includes "
        "all key details. Do not add any note.\n\n"
        f"Original:\n\"{ground_truth}\"\n\n"
        "Paraphrased version:"
    )
    return generate_response_ollama(client, prompt, "You are a helpful assistant")

def bracket_and_list_factual_data(client, a0_text):
    prompt = (
        "Take the following text and put each piece of factual data between square brackets. "
        "Then return ONLY a JSON array of these bracketed items. Example output:\n"
        "[\"[Fuel]\", \"[1983]\", \"[Mr. John Colman]\"]\n\n"
        f"Text:\n\"{a0_text}\"\n\n"
        "Output ONLY the JSON array, no extra text."
    )
    response_text = generate_response_ollama(client, prompt, "You are a helpful assistant")

    # Attempt to parse the response as JSON
    try:
        # If the response is a list of lists, flatten it
        if response_text.startswith("[[") and response_text.endswith("]]"):
            # Remove the outer brackets and split by comma
            response_text = response_text[1:-1]
            # Flatten the list and ensure each item is formatted correctly
            factual_data_list = [f"[{item.strip().strip('[]')}]"
                                 for item in response_text.split("], [")]
        else:
            factual_data_list = json.loads(response_text)
    except json.JSONDecodeError:
        print("WARNING: Could not parse JSON from model. Response was:", response_text)
        factual_data_list = []

    return factual_data_list

def filter_factual_data_by_question(factual_data, question):
    """
    Filters out any factual item that has overlapping words with the question.
    If any token in the bracketed item appears in the question, we skip that item.
    """
    filtered_data = []

    # Simple tokenization of the question
    # (strip punctuation, lowercase, then split on whitespace)
    question_tokens = set(re.findall(r"\w+", question.lower()))

    for item in factual_data:
        # Remove brackets and tokenize the item
        # e.g. "[1809]" -> "1809" -> tokens ["1809"]
        item_text = item.replace("[", "").replace("]", "")
        item_tokens = set(re.findall(r"\w+", item_text.lower()))

        # If there's NO overlap, keep it
        if question_tokens.isdisjoint(item_tokens):
            filtered_data.append(item)

    return filtered_data

def generate_noisy_version(client, a0_text, items_to_change):
    if not items_to_change:
        return a0_text  # no changes

    bracketed_items_str = ", ".join(items_to_change)
    print(f"Items to change: {bracketed_items_str}")
    promptInstructions = pathlib.Path("prompt.txt").read_text()
    prompt = (
        f"```\n{a0_text}\n```\nItems to change: {items_to_change}\nOUTPUT: "
    )

    noisy_text = generate_response_ollama(client, prompt, promptInstructions)
    return noisy_text.strip()

def main_pipeline(data):
    client = ollama.Client("http://atlas1api.eurecom.fr:8019")
    results = []

    for entry in data:
        question = entry["question"]
        print(f"Processing question: {entry['question']}")
        ground_truth = entry["ground_truth"]

        a0 = generate_a0(client, ground_truth)

        factual_data = bracket_and_list_factual_data(client, a0)

        # Filter factual data based on the question
        filtered_factual_data = filter_factual_data_by_question(factual_data, question)

        random.shuffle(filtered_factual_data)

        answers = {}
        answers["A0"] = a0
        fractions = [0.25, 0.50, 0.75, 1.0]

        for i, fraction in enumerate(fractions, start=1):
            num_items_to_change = math.ceil(len(filtered_factual_data) * fraction)
            items_for_level = filtered_factual_data[:num_items_to_change]
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