import argparse
import json
import pathlib
from typing import Dict, Any, List

import ollama

INSTRUCTIONS = (
    """
    Rewrite the provided sentence to express the same idea in a slightly different way but retain accuracy and completeness.
    Example:
    Original sentence: The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world.
    Modified sentence: The purpose of physics is to reveal universal principles, describe natural occurrences, integrate understanding, and anticipate outcomes in the physical universe.
    """,
    """
    Rewrite the sentence to introduce a minor inaccuracy or omission while keeping it mostly true.
    Example:
    Original sentence: The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world.
    Modified sentence: The purpose of physics is to reveal universal principles, describe natural occurrences, and anticipate outcomes in the physical universe.
    """,
    """
    Rewrite the sentence to include a mix of correct information from the original sentence and hallucinated information that is wrong in nuanced ways.
    Example:
    Original sentence: The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world.
    Modified sentence: The purpose of physics is to reveal universal principles, describe unexpected phenomena, provide inquestionable source of understanding, and anticipate outcomes in the physical universe.
    """,
    """
    Rewrite the sentence to make it mostly wrong in nuanced ways while retaining a small element of truth from the original sentence.
    Example:
    Original sentence: The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world.
    Modified sentence: The purpose of physics is to reveal universal principles, describe unexpected phenomena, provide inquestionable source of understanding, and predict outcomes of a metaphysical universe.
    """,
    """
    Rewrite the sentence so it contradicts in a nuanced way the original meaning.
    Example:
    Original sentence: The goal of physics is to uncover universal laws, explain natural phenomena, unify knowledge, and make predictions about the physical world.
    Modified sentence: The purpose of physics is to reveal God's universal laws, describe unexpected phenomena, provide inquestionable source of understanding, and predict outcomes of a metaphysical universe.
    """
)


def noise_pipeline(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    client = ollama.Client("http://atlas1api.eurecom.fr")

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
            "question": "What are the major branches of physics?",
            "ground_truth": "The major branches of physics include classical mechanics, electromagnetism, "
                            "thermodynamics, quantum mechanics, and relativity.",
        },
        {
            "question": "What does physics study?",
            "ground_truth": "Physics studies matter, its motion and behavior through space and time, as well as the "
                            "related entities of energy and force.",
        },
        {
            "question": "What is quantum mechanics?",
            "ground_truth": "Quantum mechanics is a branch of physics that describes the behavior of matter and energy "
                            "at very small scales, such as atoms and subatomic particles. It differs fundamentally "
                            "from classical mechanics, which governs the behavior of macroscopic objects, and "
                            "introduces a set of principles that challenge our everyday intuition about how the "
                            "world works. The key principles of quantum mechanics include the wave-particle duality, "
                            "quantization, the uncertainty principle, superposition, entanglement and measurement "
                            "limitations.",
        },
    ]

    noise_dataset = noise_pipeline(dataset)

    with open(args.output_file, "w") as f:
        json.dump(noise_dataset, f, indent=4)
