# Imports
import argparse
import json
import pathlib
from typing import Dict, Any, List

import ollama

INSTRUCTIONS = (
 """
 [Level a0 - Minimal noise]:
 Rewrite the provided sentence to express the same idea in slightly different words while preserving the full accuracy, completeness, and meaning. Do not introduce any errors or omissions.
 """,
 """
 [Level a1 - Minor inaccuracy]:
 Rewrite the sentence so that it remains mostly correct and retains its original purpose, but introduce one subtle inaccuracy or a small omission.
 """,
 """
 [Level a2 - Mixed correctness and errors ]:
 **Randomly select at least one** from the following “noise" types to introduce, but keep the sentence mostly on-topic:
 1. **Minor incorrectness**: Introduce small factual or contextual errors.
 2. **Slight contradiction**: Contradict a minor detail or nuance in the original.
 3. **Omission**: Omit a portion of the original meaning.
 """,
 """
 [Level a3 - Mostly incorrect with a kernel of truth ]:
 **Randomly select at least two** from the following “noise” types:
 1. **Medium or multiple incorrectnesses**: Introduce more obvious factual errors.
 2. **Partial contradiction**: Contradict a bigger chunk of the original meaning.
 3. **Omission**: Omit significant but not all parts of the original idea.
 """,
 """
 [Level a4 - Contradictory or fundamentally incorrect]:
 **Randomly select two or three** of the following “noise” types:
 1. **Significant incorrectness**: Replace core details with blatantly wrong or nonsensical information.
 2. **Direct contradiction**: Undermine the fundamental meaning of the original.
 3. **Major omission or twisted context**: Remove critical content or wildly shift the context.
 """
)

def load_dataset(file_path: pathlib.Path) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)

def noise_pipeline(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    client = ollama.Client("http://atlas1api.eurecom.fr:8019")

    noise_dataset = []
    for entry in dataset:
        ground_truth: str = entry["ground_truth"]
        sentences = [s + "." for s in ground_truth.split(".") if len(s.strip()) > 0]

        answers = {}
        for i, instruction in enumerate(INSTRUCTIONS):
            level_key = f"A{i}"
            output = []
            for sentence in sentences:
                response = client.chat(
                    "mistral-nemo:12b-instruct-2407-fp16",
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": f"Original sentence: {sentence}\nModified sentence: "}
                    ])

                modified = response["message"]["content"]
                output.append(modified)
            answers[level_key] = " ".join(output)

        noise_dataset.append(
            {"question": entry["question"], "ground_truth": ground_truth, "answers": answers}
        )

    return noise_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-file", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--input-file", "-i", required=True, type=pathlib.Path)

    args = parser.parse_args()

    dataset = load_dataset(args.input_file)

    noise_dataset = noise_pipeline(dataset)

    with open(args.output_file, "w") as f:
        json.dump(noise_dataset, f, indent=4)
