import argparse
import pathlib
from typing import Set, Tuple, List, Optional

import numpy as np
import ollama
import torch
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import Callbacks
from langchain_openai import ChatOpenAI
from numpy.typing import NDArray
from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness
from sentence_transformers import CrossEncoder

from models import Dataset, RagasEvaluationItem


class FastFactualCorrectness(FactualCorrectness):
    label_mapping = ['contradiction', 'entailment', 'neutral']
    nli_model: Optional[CrossEncoder] = None

    def __post_init__(self):
        super().__post_init__()

        model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-large",
            tokenizer_kwargs={"use_fast": False}
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.to(device)
        self.nli_model = model

    async def verify_claims(
            self, premise: str, hypothesis_list: List[str], callbacks: Callbacks
    ) -> NDArray[np.bool_]:
        raise NotImplementedError

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        reference = sample.reference
        response = sample.response

        response_claims = await self.decompose_claims(response, callbacks)
        reference_claims = await self.decompose_claims(reference, callbacks)

        assert self.nli_model is not None, "CrossEncoder must be loaded"
        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"

        # Compute precision
        precision_pairs = [(reference, resp) for resp in response_claims]
        precision_scores = self.nli_model.predict(precision_pairs)
        precision_labels = [self.label_mapping[score_max] for score_max in precision_scores.argmax(axis=1)]
        precision_entailments = np.array([label == "entailment" for label in precision_labels], dtype=bool)

        TP_p = np.sum(precision_entailments)
        FP_p = len(response_claims) - TP_p

        # Compute recall
        recall_pairs = [(response, ref) for ref in reference_claims]
        recall_scores = self.nli_model.predict(recall_pairs)
        recall_labels = [self.label_mapping[score_max] for score_max in recall_scores.argmax(axis=1)]
        recall_entailments = np.array([label == "entailment" for label in recall_labels], dtype=bool)

        TP_r = np.sum(recall_entailments)
        FN_r = len(reference_claims) - TP_r

        precision = TP_p / (TP_p + FP_p) if (TP_p + FP_p) > 0 else 0.0
        recall = TP_r / (TP_r + FN_r) if (TP_r + FN_r) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return np.round(f1, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["fast-fc", "default"], required=True)
    parser.add_argument("--ollama-client", required=False)
    parser.add_argument("--fact-extraction-llm", required=False)
    parser.add_argument("--input-dataset", type=pathlib.Path, required=True)
    parser.add_argument("--output-dataset", type=pathlib.Path, required=True)

    args = parser.parse_args()

    with open(args.input_dataset, "r") as f:
        dataset = Dataset.model_validate_json(f.read())

    if args.provider == "fast-fc":
        if not args.ollama_client:
            raise ValueError("Must provide --ollama-client when using Ollama provider")
        if not args.fact_extraction_llm:
            raise ValueError("Must provide --fact-extraction-llm when using Ollama provider")
        client = ollama.Client(args.ollama_client)
        client.ps()
        models = set(m["name"].split("/")[-1] for m in client.list()["models"])
        if args.fact_extraction_llm not in models:
            print(f"Pulling {args.fact_extraction_llm}")
            client.pull(args.eval_llm)
        evaluator_llm = LangchainLLMWrapper(
            Ollama(
                base_url=args.ollama_client,
                model=args.fact_extraction_llm,
                temperature=0,
                num_ctx=4096
            )
        )
        batch_size = 1
        metric = FastFactualCorrectness(atomicity="high", coverage="high", mode="f1")
        fact_extraction_llm = args.fact_extraction_llm
    else:
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-4o-mini", temperature=0))
        batch_size = 20
        metric = FactualCorrectness(atomicity="high", coverage="high", mode="f1")
        fact_extraction_llm = "gpt-4o-mini"

    client = ollama.Client(args.ollama_client)

    processed: Set[Tuple[str, int, str]] = set()
    output_file = args.output_dataset
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                item = RagasEvaluationItem.model_validate_json(line)
                processed.add((item.type, item.fact_extraction_llm, item.id, item.level))

    dataset_samples = []
    level_map = {}
    for q in dataset.questions:
        for level, a in list(q.answers.items()):

            key = (args.provider, fact_extraction_llm, q.id, level)
            if key in processed:
                continue

            dataset_samples.append({
                'user_input': q.question,
                'response': a,
                'reference': q.ground_truth,
            })
            level_map[(q.question, a)] = {
                'id': q.id,
                'user_input': q.question,
                'level': level,
                'response': a,
                'reference': q.ground_truth,
            }

    eval_dataset = EvaluationDataset.from_dict(dataset_samples)

    results = evaluate(
        dataset=eval_dataset,
        metrics=[metric],
        llm=evaluator_llm,
        batch_size=batch_size
    )

    with open(args.output_dataset, "a", encoding="UTF-8") as f_out:
        for _, row in results.to_pandas().iterrows():
            question = row["user_input"]
            ground_truth = row["reference"]
            answer = row["response"]
            score = row["factual_correctness(mode=f1)"]

            i = level_map[(question, answer)]

            eval_item = RagasEvaluationItem(
                type=args.provider,
                fact_extraction_llm=fact_extraction_llm,
                id=i["id"],
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                level=i["level"],
                result=score
            )
            f_out.write(eval_item.model_dump_json() + "\n")
            f_out.flush()
