import argparse
import json
import pathlib
from typing import Dict, Any, List

import ollama


INSTRUCTIONS = (
  """
  [a0]
  Paraphrase the sentence accurately, changing only wording but not facts.
  Example: Original: "Cats are mammals." -> "Felines belong to the class of mammals."
  """,
  """
  [a1]
  Slightly alter the sentence to remove a detail or introduce a small, subtle inaccuracy, while mostly preserving meaning.
  Example: Original: "Cats are mammals that often like milk." -> "Cats are mammals that typically enjoy milk."
  """,
  """
  [a2]
  Keep the core topic but weave in a few misleading or incorrect details, blending truth and falsehood.
  Example: Original: "Cats are mammals that often like milk." -> "Cats are creatures that belong to a mammalian order and frequently prefer sweet liquids like milk."
  """,
  """
  [a3]
  Make the sentence mostly incorrect or off-base, but still faintly related to the original subject.
  Example: Original: "Cats are mammals that often like milk." -> "Cats are amphibious reptiles commonly attracted to sugary fluids."
  """,
  """
  [a4]
  Twist the sentence so it contradicts the original meaning, while staying on the general topic.
  Example: Original: "Cats are mammals that often like milk." -> "Cats are not mammals at all and rarely have any interest in milk."
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
