import argparse
import json
import pathlib
from typing import Dict, Any, List

import ollama


INSTRUCTIONS = (
  """
  [a0 - Exact Rewording]
  Paraphrase the given sentence to convey the same facts and meaning, just with different phrasing.
  """,
  """
  [a1 - Minor Inaccuracy]
  Paraphrase the sentence but introduce one small inaccuracy or leave out a minor detail, keeping it mostly true.
  """,
  """
  [a2 - Partial Misinformation]
  Rewrite the sentence so that it contains a mix of correct and incorrect information. Stay on the same general topic, but distort some details.
  """,
  """
  [a3 - Largely Incorrect]
  Rewrite the sentence so that it is mostly wrong or misleading, yet still loosely connected to the original topic.
  """,
  """
  [a4 - Opposite Meaning]
  Rewrite the sentence so that it contradicts the original meaning or presents a fundamentally altered and incorrect view, while remaining on a related subject.
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
