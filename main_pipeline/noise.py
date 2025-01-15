# Imports
import argparse
import json
import pathlib
from typing import Dict, Any, List

import ollama

INSTRUCTIONS = (
    """
    [Level a0 - Minimal noise]:
    Rewrite the provided sentence to express the same idea in slightly different words while preserving full accuracy, completeness, and meaning. Ensure the content remains faithful to the original and includes all key details.
    Example:
    Original: "To start and propagate, a fire needs three elements: heat, fuel, and oxygen. Eliminating any of these three elements will effectively stop a fire. One can remove the fuel by cleaning the area with firebreaks (spaces cleared from vegetation), or by starting a controlled fire. Another way to stop a fire is by removing heat, typically done by pouring water and particularly effective for solid combustibles like wood or paper. The last way to stop a fire is by removing oxygen. This approach typically involves throwing at the fire materials like sand, dirt, or fire blankets."
    Rewritten: "A fire requires three key elements to ignite and spread: heat, fuel, and oxygen. Removing any of these elements can effectively extinguish a fire. One method involves eliminating the fuel, which can be done by creating firebreaks—areas cleared of vegetation—or by starting a controlled fire. Another approach is to remove the heat, typically by applying water, which is especially effective for solid fuels like wood or paper. Finally, removing oxygen is another way to stop a fire, which is often achieved by covering it with materials like sand, dirt, or fire blankets."
    """,
    """
    [Level a1 - Minor inaccuracy]:
    Rewrite the sentence so that it remains mostly correct and retains its original purpose but introduces one small omission, inaccuracy, or phrasing that subtly shifts the emphasis. Ensure the overall statement is still recognizable and logical.
    Example:
    Original: "To start and propagate, a fire needs three elements: heat, fuel, and oxygen. Eliminating any of these three elements will effectively stop a fire. One can remove the fuel by cleaning the area with firebreaks (spaces cleared from vegetation), or by starting a controlled fire. Another way to stop a fire is by removing heat, typically done by pouring water and particularly effective for solid combustibles like wood or paper. The last way to stop a fire is by removing oxygen. This approach typically involves throwing at the fire materials like sand, dirt, or fire blankets."
    Rewritten: "A fire requires three key elements to ignite and spread: heat, fuel, and air. Removing any of these elements can effectively extinguish a fire. One method involves eliminating the fuel, which can be done by creating firebreaks—areas cleared of vegetation—or by starting a controlled fire. Another approach is to remove the heat, typically by applying water, which is especially effective for solid fuels like wood or paper. Finally, removing air is another way to stop a fire, which is often achieved by covering it with materials like sand, dirt, or fire blankets."
    """,
    """
    [Level a2 - Mixed correctness and errors]:
    Rewrite the sentence to combine mostly accurate information with subtle distortions or small inaccuracies. Introduce details that could mislead or create minor confusion.
    Example:
    Original: "To start and propagate, a fire needs three elements: heat, fuel, and oxygen. Eliminating any of these three elements will effectively stop a fire. One can remove the fuel by cleaning the area with firebreaks (spaces cleared from vegetation), or by starting a controlled fire. Another way to stop a fire is by removing heat, typically done by pouring water and particularly effective for solid combustibles like wood or paper. The last way to stop a fire is by removing oxygen. This approach typically involves throwing at the fire materials like sand, dirt, or fire blankets."
    Rewritten: "A fire requires three key elements to ignite and spread: heat, wood, and air. Removing any of these elements can effectively extinguish a fire. One method involves eliminating the wood, which can be done by clearing regions of risk from vegetation. Another approach is to remove the heat, typically by applying water, which is especially effective for solid fuels like wood or paper. Finally, removing air is another way to stop a fire, which is often achieved by covering it with materials like sand, dirt, or fire blankets."
    """,
    """
    [Level a3 - Distorted Information]:
    Rewrite the sentence so that most of the information is incorrect, misleading, or distorted while maintaining a faint connection to the original topic. Include at least two elements that reflect a recognizable truth.
    Example:
    Original: "To start and propagate, a fire needs three elements: heat, fuel, and oxygen. Eliminating any of these three elements will effectively stop a fire. One can remove the fuel by cleaning the area with firebreaks (spaces cleared from vegetation), or by starting a controlled fire. Another way to stop a fire is by removing heat, typically done by pouring water and particularly effective for solid combustibles like wood or paper. The last way to stop a fire is by removing oxygen. This approach typically involves throwing at the fire materials like sand, dirt, or fire blankets."
    Rewritten: "A fire requires three key elements to ignite and spread: heat, wood, and air. Removing any of these elements can effectively extinguish a fire. One method involves eliminating the wood, which can be done by clearing regions of risk from vegetation. Another approach is to remove the heat, typically by applying water, which is especially effective for liquid fuels like gasoline or oil. Finally, removing air is another way to stop a fire, which is often achieved by covering it with materials like sand, dirt, or fire blankets."
    """,
    """
    [Level a4 - Fundamentally incorrect and Fabricated Information]:
    Rewrite the sentence so that it fundamentally misrepresents the event by introducing fabricated or non-existent details. Ensure the topic appears related, but alter the intent or core message to render it fundamentally inaccurate.
    Example:
    Original: "To start and propagate, a fire needs three elements: heat, fuel, and oxygen. Eliminating any of these three elements will effectively stop a fire. One can remove the fuel by cleaning the area with firebreaks (spaces cleared from vegetation), or by starting a controlled fire. Another way to stop a fire is by removing heat, typically done by pouring water and particularly effective for solid combustibles like wood or paper. The last way to stop a fire is by removing oxygen. This approach typically involves throwing at the fire materials like sand, dirt, or fire blankets."
    Rewritten: "A fire requires three key elements to ignite and spread: fire, wood, and air. Removing any of these elements may extinguish a fire. One method involves eliminating the wood, which can be done by cutting down trees. Another approach is to remove the original fire by throwing water at it. Finally, removing air is another way to stop a fire, but this is very hard to achieve since living organisms need it for living."
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
