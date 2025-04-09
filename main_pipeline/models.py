from typing import Dict, Optional, List

import pydantic


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


class AssessmentItem(pydantic.BaseModel):
    id: int
    question: str
    ground_truth: str
    answers: Dict[str, Dict[str, str]]


class AssessmentDataset(pydantic.BaseModel):
    questions: List[AssessmentItem]
