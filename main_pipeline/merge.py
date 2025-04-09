import argparse
import pathlib
from typing import Iterable

from main_pipeline.models import Dataset, Item, AssessmentItem, AssessmentDataset


def inner_join(list_i: Iterable[Item], list_j: Iterable[Item]):
    # Create dictionaries mapping IDs to their full tuples
    dict1 = {item.id: item for item in list_i}
    dict2 = {item.id: item for item in list_j}

    # Find IDs present in both lists
    common_ids = dict1.keys() & dict2.keys()

    # Return pairs of tuples with matching IDs (sorted by ID)
    return [(dict1[id], dict2[id]) for id in sorted(common_ids)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-dataset-path", type=pathlib.Path, required=True)
    parser.add_argument("--ai-dataset-path", type=pathlib.Path, required=True)
    parser.add_argument("--assessment-dataset-dir", type=pathlib.Path, required=True)

    args = parser.parse_args()

    with open(args.human_dataset_path, "r") as f:
        human_dataset = Dataset.model_validate_json(f.read())

    with open(args.ai_dataset_path, "r") as f:
        ai_dataset = Dataset.model_validate_json(f.read())

    items = []
    for human, ai in inner_join(human_dataset.questions, ai_dataset.questions):
        if human.question != ai.question:
            raise ValueError(f"Questions are not the same, are you sure about this? {human.question} / {ai.question}")

        if human.ground_truth != ai.ground_truth:
            raise ValueError(
                f"Ground Truths are not the same, are you sure about this? {human.ground_truth} / {ai.ground_truth}"
            )

        answers = {
            k: {"ai": a, "human": h}
            for (_, h), (k, a)
            in zip(sorted(human.answers.items()), sorted(ai.answers.items()))
        }
        items.append(
            AssessmentItem(id=human.id, question=human.question, ground_truth=human.ground_truth, answers=answers)
        )

    with open(args.assessment_dataset_dir / "assessment-dataset.json", "w") as f:
        f.write(AssessmentDataset(questions=items).model_dump_json(indent=4))
