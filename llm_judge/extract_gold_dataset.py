import argparse
import gzip
import html
import json
import pathlib
import re
from typing import Dict, List

import numpy as np
from bs4 import BeautifulSoup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("google_nq_dataset", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--target-size", type=int, default=200)
    args = parser.parse_args()

    gold_dataset: List[Dict[str, str]] = []

    with gzip.open(args.google_nq_dataset) as f:
        for line in f:
            item = json.loads(line)

            if sum([annotation["long_answer"]["start_byte"] >= 0 for annotation in item["annotations"]]) < 2:
                # item does not have long answer
                continue

            question = item["question_text"] + "?"
            long_answers = [a["long_answer"] for a in item["annotations"]]
            long_answer_bounds = [
                (la["start_byte"], la["end_byte"]) for la in long_answers
            ]
            long_answer_counts = [
                long_answer_bounds.count(la) for la in long_answer_bounds
            ]
            long_answer = long_answers[np.argmax(long_answer_counts)]
            start, end = long_answer["start_byte"], long_answer["end_byte"]
            document_html = item["document_html"].encode("utf-8")
            long_answer_html = document_html[start:end].decode()

            soup = BeautifulSoup(long_answer_html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            decoded_text = html.unescape(text)
            clean_extra_spaces = re.sub(r"\s+([.,;:!?)-])", r"\1", decoded_text)
            clean_extra_spaces = re.sub(r"\(\s+", r"(", clean_extra_spaces)
            clean_citations = re.sub(r'\s*\[\d+]', '', clean_extra_spaces)

            gold_answer = clean_citations.strip()

            if len(gold_answer) < 250 or len(long_answer) > 700 or not gold_answer.endswith("."):
                continue

            gold_dataset.append({"question": question, "ground_truth": gold_answer})

            if len(gold_dataset) >= args.target_size:
                break

    with open(args.output_path, "w") as f:
        f.write(json.dumps(gold_dataset, indent=4))
