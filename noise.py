import argparse
import json
import pathlib
from typing import Dict, Any, List

import ollama


INSTRUCTIONS = (
  """
  [Level a0 - Minimal noise]:
  Rewrite the provided sentence to express the same idea in slightly different words while preserving the full accuracy, completeness, and meaning. Do not introduce any errors or omissions.
  Example:
  Original: "The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world."
  Rewritten: "The purpose of physics is to reveal universal principles, describe natural phenomena, integrate understanding, and predict outcomes in the physical world."
  """,
  """
  [Level a1 - Minor inaccuracy]:
  Rewrite the sentence so that it remains mostly correct and retains its original purpose, but introduce one subtle inaccuracy or a small omission. Ensure the overall statement still closely aligns with the original meaning, but slightly reduces accuracy.
  Example:
  Original: "The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world."
  Rewritten: "The purpose of physics is to reveal universal principles, describe natural occurrences, and predict outcomes in the physical world."
  (Here, the mention of 'unify knowledge' is omitted, slightly reducing completeness.)
  """,
  """
  [Level a2 - Mixed correctness and errors]:
  Rewrite the sentence to blend correct information from the original with some incorrect or misleading details. The statement should still be recognizable and mostly on-topic, but contain multiple subtle errors or distortions.
  Example:
  Original: "The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world."
  Rewritten: "The purpose of physics is to reveal universal principles, describe unexpected phenomena, provide an unquestioned source of understanding, and anticipate possibilities in the physical realm."
  (Here, phrases like 'unquestioned source of understanding' are misleading and overly confident, subtly distorting the meaning.)
  """,
  """
  [Level a3 - Mostly incorrect with a kernel of truth]:
  Rewrite the sentence so that most of the information is incorrect, misleading, or off-base, yet still loosely connected to the original topic. Include at least one small element that remains truthful or recognizable, but let the majority of the details be wrong in nuanced ways.
  Example:
  Original: "The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world."
  Rewritten: "The aim of physics is to uncover mysterious doctrines, interpret strange illusions, provide unquestionable beliefs, and foretell events in a distant, metaphysical universe."
  (There’s a faint connection to seeking fundamental truths, but the rest is clearly off-topic or incorrect.)
  """,
  """
  [Level a4 - Contradictory or fundamentally incorrect]:
  Rewrite the sentence so that it contradicts or undermines the original meaning in a nuanced but clear manner. The topic should still be related, but the core message should now be fundamentally altered or false.
  Example:
  Original: "The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world."
  Rewritten: "The purpose of physics is to reveal divine decrees, interpret mystical occurrences, consolidate faith-based wisdom, and predict outcomes in a supernatural cosmos."
  (This version shifts from a scientific perspective to a religious/mystical one, effectively contradicting the original intent.)
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
