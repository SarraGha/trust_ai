# -*- coding: utf-8 -*-
import abc
import argparse
import json
import pathlib
import random
import re
from typing import List, Dict, Tuple, Set, Union, Iterator

import spacy
from openai import OpenAI
from spacy import Language, Errors
from spacy.lang.en import stop_words
from spacy.symbols import NOUN, PROPN, ADV, ADJ, amod
from spacy.tokens import Doc, Span
from tqdm import tqdm

from models import Sample, Tracker, Report


def process_terms(text: str, allowed_terms: Set[str]) -> str:
    # Function to determine replacement for each matched bracketed term
    def replacer(match):
        term = match.group(1)
        return f'[{term}]' if term in allowed_terms else f'{{{{{term}}}}}'

    # Use regex to find all bracketed terms and apply the replacer function
    processed_text = re.sub(r'\[([^]]+)]', replacer, text)
    return processed_text


class LLM:

    def __init__(self, client: OpenAI):
        self._client = client

    def query(self, messages: List[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(model="gpt-4o", messages=messages)
        return completion.choices[0].message.content.strip()


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
        return collected, tracker


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

    def __init__(self, nlp: Language):
        self.nlp = nlp

    @classmethod
    def span_boxes(cls, doclike: Union[Doc, Span]) -> Iterator[Span]:
        """
        Detect base noun phrases in the object and adverbs from a dependency parse.
        """
        labels = [
            "oprd",
            "dobj",
            "advmod",
            "amod",
            "npadvmod",
            "pcomp",
            "pobj",
            "dative",
            "appos",
            "attr",
            "ROOT",
        ]
        doc = doclike.doc  # Ensure works on both Doc and Span.
        if not doc.has_annotation("DEP"):
            raise ValueError(Errors.E029)
        np_deps = [doc.vocab.strings.add(label) for label in labels]
        conj = doc.vocab.strings.add("conj")
        prev_end = -1

        # Collect subject heads within the doclike (Span or Doc)
        subject_heads = [token for token in doclike if token.dep_ in {'nsubj', 'nsubjpass'}]

        # Collect indices of all tokens in their subtrees
        subject_indices = set()
        for head in subject_heads:
            subject_indices.update(t.i for t in head.subtree)

        for i, word in enumerate(doclike):
            if word.pos not in (NOUN, PROPN, ADV, ADJ):
                continue

            # Skip if part of the subject
            if word.i in subject_indices:
                continue

            if word.pos == ADJ and word.dep == amod:
                continue

            # Prevent nested chunks from being produced
            if word.left_edge.i <= prev_end:
                continue
            if word.dep in np_deps:
                prev_end = word.i
                yield doc[word.left_edge.i:word.i + 1]
            elif word.dep == conj:
                head = word.head
                while head.dep == conj and head.head.i < head.i:
                    head = head.head
                # If the head is an NP, and we're coordinated to it, we're an NP
                if head.dep in np_deps:
                    prev_end = word.i
                    yield doc[word.left_edge.i:word.i + 1]

    @classmethod
    def overlaps(cls, idx: List[Tuple[int, int]]) -> bool:
        sorted_intervals = sorted(idx)
        return any(
            current_end > next_start
            for (_, current_end), (next_start, _) in zip(sorted_intervals, sorted_intervals[1:])
        )

    def tag_predicate_nouns_and_adverbs(self, sentence: str):
        doc = self.nlp(sentence)

        idx = []
        for box in FactualDataStep.span_boxes(doc):
            idx.append((min(b.idx for b in box), max(b.idx + len(b) for b in box)))

        idx.sort(reverse=True)

        assert not FactualDataStep.overlaps(idx), f"Something went wrong... Overlapping indexes for `{sentence}`"

        boxed_sentence = sentence
        for start, end in idx:
            boxed_sentence = boxed_sentence[:start] + "[" + boxed_sentence[start:end] + "]" + boxed_sentence[end:]

        return boxed_sentence

    def step(self, sample: Sample, tracker: Tracker) -> None:
        assert "A0" in sample.answers.keys()

        response_text = self.tag_predicate_nouns_and_adverbs(sample.answers["A0"])
        sample.with_brackets["A0"] = response_text

        matches = re.findall(r"\[(.*?)]", response_text)
        if not matches:
            tracker.find_factual_data_error += 1
            return

        sample.raw_factual_data = matches


class FilterItemsFromQuestionStep(Step):
    """
    Filters out any factual item that has overlapping words with the question.
    If any token in the bracketed item appears in the question, we skip that item.
    """

    def __init__(self):
        super().__init__()
        self._stop_words = set(stop_words.STOP_WORDS)

    def step(self, sample: Sample, tracker: Tracker) -> None:
        if not sample.is_initialized():
            return

        assert sample.question is not None
        assert sample.raw_factual_data is not None
        assert len(sample.raw_factual_data) > 0

        # Simple tokenization of the question
        # (strip punctuation, lowercase, then split on whitespace)
        question_words = set(re.findall(r"\w+", sample.question.lower()))
        question_words = question_words - self._stop_words
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

    @classmethod
    def split_groups(cls, idx: List[int], num: int) -> List[List[int]]:
        group_size = len(idx) // num
        remainder = len(idx) % num
        groups = []
        start = 0
        for i in range(num):
            current_size = group_size + 1 if i < remainder else group_size
            groups.append(idx[start:start + current_size])
            start += current_size
        return groups

    def step(self, sample: Sample, tracker: Tracker) -> None:
        if not sample.is_initialized():
            return

        idx = list(range(len(sample.factual_data)))
        random.shuffle(idx)
        groups = CreateNoiseExamplesStep.split_groups(idx, self._levels)
        a0 = sample.with_brackets["A0"]
        noised_sample = a0
        for i, group in enumerate(groups, start=1):
            selected = [sample.factual_data[j] for j in group]
            formatted_list = [f"[{term}]" for term in selected]
            items_to_change = ', '.join(formatted_list)

            input_sample = process_terms(noised_sample, items_to_change)

            prompt = (
                f"```\n{input_sample}\n```\n\nOUTPUT: "
            )

            output_sample = self._llm.query(
                [
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            noised_sample = re.sub(r'\{\{(.*?)}}', r'[\1]', output_sample)
            sample.with_brackets[f"A{i}"] = noised_sample
            cleaned = re.sub(r'\{\{(.*?)}}', r'\1', output_sample)
            sample.answers[f"A{i}"] = cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--input-file", "-i", required=True, type=pathlib.Path)

    args = parser.parse_args()

    client = OpenAI()
    llm = LLM(client)

    nlp = spacy.load("en_core_web_sm")

    pipeline = (
        Pipeline()
        .with_step(ParaphraseStep(llm))
        .with_step(FactualDataStep(nlp))
        .with_step(FilterItemsFromQuestionStep())
        .with_step(CreateNoiseExamplesStep(llm))
    )

    samples, tracker = pipeline.run(JsonReader(args.input_file))

    report = Report(report=tracker, questions=samples)
    dataset = report.to_dataset()

    with open(args.output_dir / "report.json", "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=4))

    with open(args.output_dir / "dataset.json", "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=4))

    print()
    print("Stats")
    print(tracker.model_dump_json(indent=4))
