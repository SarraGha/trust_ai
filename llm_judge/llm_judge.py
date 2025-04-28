import abc
import argparse
import pathlib
from typing import Optional, Tuple, Set

import ollama
from tqdm import tqdm
from transformers import AutoTokenizer

from models import Dataset, LLMasJudgeEvaluationItem

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{question}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
Does the response demonstrate factual correctness by covering all essential points from the the reference answer without introducing inaccuracies, omissions, or hallucinated information?

- Score 1: The response is mostly factually incorrect or misleading, with many inaccuracies or fabricated information.
- Score 2: The response contains multiple factual inaccuracies, significant omissions, or introduces misleading/untrue statements that affect the answer quality.
- Score 3: The response is mostly factually correct, but has some inaccuracies, omissions, or unsupported information that weakens it.
- Score 4: The response is factually correct and mostly complete, with very minor omissions or imprecisions that do not affect the overall understanding.
- Score 5: The response is entirely factually accurate, fully complete based on the instruction, and does not introduce any inaccuracies, fabrications, or unsupported information.


###Feedback: """


class Evaluator(abc.ABC):

    def __init__(self, max_retry: int = 5):
        self._max_retry = max_retry

    @abc.abstractmethod
    def _query(self, prompt: str) -> str:
        ...

    def evaluate(self, prompt: str) -> Optional[Tuple[str, int]]:
        for _ in range(self._max_retry):
            try:
                evaluation = self._query(prompt)

                if "[RESULT]" not in evaluation:
                    continue

                assessment, result = evaluation.split("[RESULT]")

                assessment = assessment.strip()
                result = result.strip()

                if not result.isdigit():
                    continue

                return assessment, int(result)
            except Exception:
                continue
        return


class OllamaEvaluator(Evaluator):

    def __init__(self, client: ollama.Client, model: str, max_retry: int = 5):
        super().__init__(max_retry)
        self.client = client
        self.model = model

    def _query(self, prompt: str) -> str:
        response = client.chat(
            self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            options={
                "num_ctx": 4098,
                "temperature": 0.1,
                "seed": 0,
            }
        )
        return response["message"]["content"].strip()


class PrometheusEvaluator(Evaluator):

    def __init__(self, client: ollama.Client, max_retry: int = 5):
        super().__init__(max_retry)
        self.client = client
        self.model = "vicgalle/prometheus-7b-v2.0"
        self.tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")

    def _query(self, prompt: str) -> str:
        chat = self.tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ], tokenize=False)

        response = client.generate(
            model=self.model,
            prompt=chat,
            options={
                "num_ctx": 4098,
                "temperature": 0.1,
                "seed": 0,
            })
        return response["response"].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-llm", nargs="*", required=True)
    parser.add_argument("--ollama-client", required=True)
    parser.add_argument("--input-dataset", type=pathlib.Path, required=True)
    parser.add_argument("--output-dataset", type=pathlib.Path, required=True)

    args = parser.parse_args()

    with open(args.input_dataset, "r") as f:
        dataset = Dataset.model_validate_json(f.read())

    client = ollama.Client(args.ollama_client)

    client.ps()
    models = set(m["name"].split("/")[-1] for m in client.list()["models"])

    for model_name in args.eval_llm:
        if model_name == "prometheus":
            if "prometheus-7b-v2.0" not in models:
                print(f"Pulling vicgalle/prometheus-7b-v2.0")
                client.pull("vicgalle/prometheus-7b-v2.0")
                continue

        if model_name not in models:
            print(f"Pulling {model_name}")
            client.pull(model_name)

    processed: Set[Tuple[str, int, str]] = set()
    output_file = args.output_dataset
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                item = LLMasJudgeEvaluationItem.model_validate_json(line)
                processed.add((item.llm, item.id, item.level))

    with open(output_file, "a") as f_out:
        for model_name in tqdm(args.eval_llm, desc="Models"):
            if model_name == "prometheus":
                evaluator = PrometheusEvaluator(client)
            else:
                evaluator = OllamaEvaluator(client, model_name)

            for entry in tqdm(dataset.questions, desc=f"Evaluating {model_name}"):
                question_text = entry.question
                ground_truth = entry.ground_truth
                answers = entry.answers

                for level, response in answers.items():
                    key = (model_name, entry.id, level)
                    if key in processed:
                        continue

                    prompt = ABSOLUTE_PROMPT.format(
                        question=question_text, response=response, reference_answer=ground_truth
                    )

                    assessment = evaluator.evaluate(prompt)

                    if assessment is None:
                        continue

                    feedback, result = assessment

                    item = LLMasJudgeEvaluationItem(
                        llm=model_name,
                        id=entry.id,
                        question=question_text,
                        answer=response,
                        ground_truth=ground_truth,
                        level=level,
                        assessment=feedback,
                        result=result
                    )

                    f_out.write(item.model_dump_json() + "\n")
                    f_out.flush()
