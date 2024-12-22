import argparse
import json
import pathlib
from typing import Dict, Any, List

import ollama


INSTRUCTIONS = (
  """
  (a0) Perfect Paraphrase:
  Re-express the sentence in new words without losing or altering any facts or meaning.
  """,
  """
  (a1) Slight Distortion:
  Rewrite the sentence so that it mostly preserves the original meaning, but contains one subtle loss of detail or minor inaccuracy.
  """,
  """
  (a2) Mixed Truth and Error:
  Rewrite the sentence blending correct points with a few small but noticeable errors, yet keep the overall context recognizable.
  """,
  """
  (a3) Predominantly Wrong:
  Rewrite the sentence so that it is mostly incorrect, but retains a vague hint of the original topic or fact.
  """,
  """
  (a4) Contradiction:
  Rewrite the sentence so that it contradicts or reverses the original meaning, while staying somewhat related to the initial subject.
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
