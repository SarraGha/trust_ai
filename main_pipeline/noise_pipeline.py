# -*- coding: utf-8 -*-
import abc
import argparse
import json
import math
import pathlib
import random
import re
from typing import List, Dict, Optional, Tuple

import pydantic
from openai import OpenAI
from tqdm import tqdm


class LLM:

    def __init__(self, client: OpenAI):
        self._client = client

    def query(self, messages: List[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(model="gpt-4o", messages=messages)
        return completion.choices[0].message.content.strip()


class Sample(pydantic.BaseModel):
    question: Optional[str] = None
    ground_truth: Optional[str] = None
    raw_factual_data: Optional[List[str]] = {}
    with_brackets: Dict[str, str] = {}
    blacklisted: Optional[List[str]] = None
    factual_data: Optional[List[str]] = None
    answers: Dict[str, str] = {}

    def is_initialized(self) -> bool:
        return (
                self.question is not None and
                self.ground_truth is not None and
                "A0" in self.answers.keys() and
                self.with_brackets and
                self.raw_factual_data is not None and
                len(self.raw_factual_data) > 0
        )

    def is_valid(self) -> bool:
        return (
                self.question is not None and
                self.ground_truth is not None and
                self.with_brackets and
                self.raw_factual_data is not None and
                len(self.raw_factual_data) > 0 and
                self.blacklisted is not None and
                self.factual_data is not None and
                len(self.factual_data) > 0 and
                self.answers.keys() == {"A0", "A1", "A2", "A3", "A4"}
        )


class Tracker(pydantic.BaseModel):
    input_samples: int = 0
    find_factual_data_error: int = 0
    output_samples: int = 0


class Item(pydantic.BaseModel):
    id: int
    question: str
    ground_truth: str
    answers: Dict[str, str]

    @classmethod
    def from_sample(cls, id_: int, sample: Sample) -> 'Item':
        return Item(id=id_, question=sample.question, ground_truth=sample.ground_truth, answers=sample.answers)


class Dataset(pydantic.BaseModel):
    questions: List[Item]


class Report(pydantic.BaseModel):
    report: Tracker
    questions: List[Sample]

    def to_dataset(self) -> Dataset:
        items = [Item.from_sample(id_=i, sample=s) for i, s in enumerate(self.questions) if s.is_valid()]
        return Dataset(questions=items)


class Step(abc.ABC):

    @abc.abstractmethod
    def step(self, sample: Sample, tracker: Tracker) -> None:
        ...


class Reader(abc.ABC):
    @abc.abstractmethod
    def samples(self) -> List[Sample]:
        ...


class JsonReader(Reader):

    def __init__(self, input_file: pathlib.Path):
        self._input_file = input_file

    def samples(self) -> List[Sample]:
        with open(self._input_file, "r") as f:
            content = f.read()

        gold_dataset = json.loads(content)
        return [Sample(question=d["question"], ground_truth=d["ground_truth"]) for d in gold_dataset]


class Pipeline:

    def __init__(self):
        self._steps: List[Step] = []

    def with_step(self, step: Step) -> 'Pipeline':
        self._steps.append(step)
        return self

    def run(self, reader: Reader) -> Tuple[List[Sample], Tracker]:
        tracker = Tracker()
        collected = []
        for sample in tqdm(reader.samples(), desc="Samples:"):
            tracker.input_samples += 1
            for step in self._steps:
                step.step(sample, tracker)
            collected.append(sample)
            if sample.is_valid():
                tracker.output_samples += 1
        return tracker


class ParaphraseStep(Step):
    prompt = (
        "Rewrite the provided sentence to express the same idea in slightly different words while preserving "
        "full accuracy, completeness, and meaning. Ensure the content remains faithful to the original and includes "
        "all key details. Do not add any note.\n\n"
        "Original:\n{ground_truth}\n\n"
        "Paraphrased version:"
    )

    def __init__(self, llm: LLM):
        self._llm = llm

    def step(self, sample: Sample, tracker: Tracker) -> None:
        assert sample.ground_truth is not None
        prompt = self.prompt.format(ground_truth=sample.ground_truth)
        paraphrased = self._llm.query([{"role": "user", "content": prompt}])
        sample.answers["A0"] = paraphrased


class FactualDataStep(Step):
    prompt = (
        "Given a text extract, place a factual data between brackets [ ]. \n"
        "A fact is a piece of information that is objectively true, measurable, or verifiable. This includes:\n"
        "- Dates (e.g., [July 29th], [1991])\n"
        "- Names of people, places, and organizations (e.g., [Greek], [Pepsi])\n"
        "- Scientific and technical terms (e.g., [greenhouse gases])\n"
        "- Numerical data or direct measurements (e.g., [20%], [one third])\n"
        "- Objective actions and events that have clear historical, scientific, or legal verification "
        "(e.g., [discovered], [signed into law])\n"
        "- Well-established causal relationships, meaning causes or major contributors that are widely accepted "
        "and backed by evidence (e.g., [Deforestation] contributes to [habitat loss])\n"
        "This excludes opinions, interpretations, or vague descriptions.\n"
        "Example: \n"
        "Input: On September 6, 1609, only five days after the arrival of the first Dutch and English "
        "sailors, John Colman was reportedly killed by attacking Native Americans by an arrow to his neck."
        "\n"
        "Output: On [September 6], [1609], only [five days] after the arrival of the first [Dutch] and [English] "
        "[sailors], John Colman was [killed by] attacking [Native Americans] [by an arrow] to his [neck]."
        "\n\n"
        "Output a text with brackets around factual data with no extra annotations, formatting, or comments."
        "\n\n"
        "Input: {a0_text}"
        "\n"
        "Output: "
    )

    def __init__(self, llm: LLM):
        self._llm = llm

    def step(self, sample: Sample, tracker: Tracker) -> None:
        assert "A0" in sample.answers.keys()
        prompt = self.prompt.format(a0_text=sample.answers["A0"])

        response_text = self._llm.query([{"role": "user", "content": prompt}])
        sample.with_brackets["A0"] = response_text

        matches = re.findall(r"\[(.*?)]", response_text)
        if not matches:
            tracker.find_factual_data_error += 1
            return

        sample.raw_factual_data = matches

        cleaned = re.sub(r'\[(.*?)]', r'\1', response_text)
        sample.answers["A0"] = cleaned


class FilterItemsFromQuestionStep(Step):
    """
    Filters out any factual item that has overlapping words with the question.
    If any token in the bracketed item appears in the question, we skip that item.
    """

    def step(self, sample: Sample, tracker: Tracker) -> None:
        if not sample.is_initialized():
            return

        assert sample.question is not None
        assert sample.raw_factual_data is not None
        assert len(sample.raw_factual_data) > 0

        # Simple tokenization of the question
        # (strip punctuation, lowercase, then split on whitespace)
        question_words = set(re.findall(r"\w+", sample.question.lower()))
        sample.blacklisted = [
            term.lower() for term in sample.raw_factual_data
            if any(word.lower() in question_words for word in term.split())
        ]
        sample.factual_data = [i for i in sample.raw_factual_data if i.lower() not in sample.blacklisted]


class CreateNoiseExamplesStep(Step):

    def __init__(self, llm: LLM):
        self._llm = llm
        self._prompt = pathlib.Path("prompt.txt").read_text()
        self._levels = 4

    def step(self, sample: Sample, tracker: Tracker) -> None:
        if not sample.is_initialized():
            return

        idx = list(range(len(sample.factual_data)))
        random.shuffle(idx)
        group_size = math.ceil(len(idx) / self._levels)
        groups = [idx[i:i + group_size] for i in range(0, len(idx), group_size)]
        noised_sample = sample.with_brackets["A0"]
        for i, group in enumerate(groups, start=1):
            selected = [sample.factual_data[j] for j in group]
            formatted_list = [f"[{term}]" for term in selected]
            items_to_change = ', '.join(formatted_list)

            prompt = (
                f"```\n{noised_sample}\n```\nItems to change: {items_to_change}\nOUTPUT: "
            )

            noised_sample = self._llm.query(
                [
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            sample.with_brackets[f"A{i}"] = noised_sample
            cleaned = re.sub(r'\[(.*?)]', r'\1', noised_sample)
            sample.answers[f"A{i}"] = cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--input-file", "-i", required=True, type=pathlib.Path)

    args = parser.parse_args()

    client = OpenAI()
    llm = LLM(client)

    pipeline = (
        Pipeline()
        .with_step(ParaphraseStep(llm))
        .with_step(FactualDataStep(llm))
        .with_step(FilterItemsFromQuestionStep())
        .with_step(CreateNoiseExamplesStep(llm))
    )

    samples, tracker = pipeline.run(JsonReader(args.input_file))

    report = Report(report=tracker, questions=samples)
    dataset = report.to_dataset()

    with open(args.output_file / "report.json", "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=4))

    with open(args.output_file / "dataset.json", "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=4))

    print()
    print("Stats")
    print(tracker.model_dump_json(indent=4))
