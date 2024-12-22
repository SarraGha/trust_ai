import argparse
import json
import pathlib
from typing import Dict, Any, List

import ollama


INSTRUCTIONS = (
  """
  [a0 - Faithful Rewriting]
  Rewrite the given sentence to say the same thing in different words without adding errors or changing its meaning.
  """,
  """
  [a1 - Minor Deviation]
  Rewrite the given sentence so that it still mostly conveys the original message but introduces a small omission or slight inaccuracy.
  """,
  """
  [a2 - Noticeable Errors]
  Rewrite the given sentence by mixing correct information with a few subtle inaccuracies or distortions. The overall topic should remain recognizable, but truth should be partially compromised.
  """,
  """
  [a3 - Mostly Incorrect]
  Rewrite the given sentence so that it is mostly incorrect or misleading, but still loosely related to the original topic. Keep a small trace of the original meaning or context.
  """,
  """
  [a4 - Contradictory]
  Rewrite the given sentence so that it fundamentally contradicts or reverses the original meaning, while remaining on a related topic.
  """
)


def noise_pipeline(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    client = ollama.Client("http://atlas1api.eurecom.fr:8019")

    noise_dataset = []
    for entry in dataset:
        ground_truth: str = entry["ground_truth"]
        sentences = [s + "." for s in ground_truth.split(".") if len(s.strip()) > 0]

        for instruction in INSTRUCTIONS:
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

            noise_dataset.append(
                {"question": entry["question"], "ground_truth": ground_truth, "answer": " ".join(output)}
            )

    return noise_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-file", "-o", required=True, type=pathlib.Path)

    args = parser.parse_args()

    dataset = [
        {
            "question": "What are the main causes of climate change?",
            "ground_truth": "Climate change is primarily caused by human activities "
                            "that increase the concentration of greenhouse gases in the atmosphere. "
                            "The burning of fossil fuels (coal, oil, and natural gas) for energy and transportation "
                            "is the largest contributor. Deforestation, industrial processes, and agricultural practices "
                            "also release significant amounts of carbon dioxide, methane, and nitrous oxide, which trap "
                            "heat in the atmosphere and lead to global warming.",
        },
    ]

    noise_dataset = noise_pipeline(dataset)

    with open(args.output_file, "w") as f:
        json.dump(noise_dataset, f, indent=4)
