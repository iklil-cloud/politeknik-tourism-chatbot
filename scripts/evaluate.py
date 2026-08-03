from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHAPTERS_DIR, EVAL_DATA_PATH
from app.judge_evaluation import JUDGE_SCORE_FIELDS, LLMJudge
from app.rag_pipeline import SejarahRAG


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def score_prediction(reference: str, prediction: str) -> dict[str, float]:
    reference_tokens = tokenize(reference)
    prediction_tokens = tokenize(prediction)

    if not reference_tokens and not prediction_tokens:
        return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    common = Counter(reference_tokens) & Counter(prediction_tokens)
    overlap = sum(common.values())

    precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = 1.0 if normalize(reference) == normalize(prediction) else 0.0
    return {
        "accuracy": accuracy,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chatbot answers with token overlap, optional BERTScore, and optional LLM-as-judge."
    )
    parser.add_argument(
        "--bert-model",
        default="bert-base-multilingual-cased",
        help="Hugging Face model name for BERTScore.",
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="Batch size used when computing BERTScore.",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip BERTScore calculation.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-a-judge scoring through Ollama.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Ollama model used as judge. Defaults to the configured answer model.",
    )
    args = parser.parse_args()

    if not EVAL_DATA_PATH.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_DATA_PATH}")

    rag = SejarahRAG(CHAPTERS_DIR)
    rag.ingest(reset=False)

    with EVAL_DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        input_rows = list(csv.DictReader(handle))

    total_questions = len(input_rows)
    if total_questions == 0:
        raise ValueError(f"No questions found in {EVAL_DATA_PATH}")

    print(f"Starting evaluation for {total_questions} question(s)...")

    rows = []
    references: list[str] = []
    predictions: list[str] = []
    judge = LLMJudge(model=args.judge_model) if args.judge_model else LLMJudge()
    for index, row in enumerate(input_rows, start=1):
        question = row["question"]
        reference = row["reference_answer"]

        print(f"[{index}/{total_questions}] Evaluating: {question[:80]}")
        result = rag.answer_question(question)
        prediction = str(result["answer"])
        scores = score_prediction(reference, prediction)
        judge_scores = {}
        if args.judge:
            print(f"[{index}/{total_questions}] Running LLM judge...")
            judge_scores = judge.evaluate(question, reference, prediction).to_row()
        references.append(reference)
        predictions.append(prediction)
        rows.append(
            {
                "question": question,
                "reference_answer": reference,
                "prediction": prediction,
                **scores,
                **judge_scores,
            }
        )

    if args.skip_bert:
        for row in rows:
            row["bert_precision"] = ""
            row["bert_recall"] = ""
            row["bert_f1"] = ""
    else:
        try:
            from bert_score import score as bert_score
        except ImportError as exc:
            raise RuntimeError(
                "BERTScore is not installed. Install `bert-score` or rerun with `--skip-bert`."
            ) from exc

        print(f"Computing BERTScore with model `{args.bert_model}`...")
        bert_precision, bert_recall, bert_f1 = bert_score(
            predictions,
            references,
            model_type=args.bert_model,
            batch_size=args.batch_size,
            verbose=True,
        )

        for index, row in enumerate(rows):
            row["bert_precision"] = float(bert_precision[index].item())
            row["bert_recall"] = float(bert_recall[index].item())
            row["bert_f1"] = float(bert_f1[index].item())

    output_path = Path("results")
    output_path.mkdir(exist_ok=True)
    report = output_path / "evaluation_report.csv"

    with report.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "question",
            "reference_answer",
            "prediction",
            "accuracy",
            "token_precision",
            "token_recall",
            "token_f1",
            "bert_precision",
            "bert_recall",
            "bert_f1",
        ]
        if args.judge:
            fieldnames.extend(
                [
                    *(f"judge_{field}" for field in JUDGE_SCORE_FIELDS),
                    "judge_reason",
                    "judge_raw_response",
                ]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "accuracy",
        "token_precision",
        "token_recall",
        "token_f1",
    ]
    if not args.skip_bert:
        metric_keys.extend(["bert_precision", "bert_recall", "bert_f1"])
    if args.judge:
        metric_keys.extend(f"judge_{field}" for field in JUDGE_SCORE_FIELDS)

    metrics = {key: sum(float(row[key]) for row in rows) / len(rows) for key in metric_keys}
    print("Evaluation complete.")
    print("Average metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.3f}")
    print(f"Saved report to {report}")


if __name__ == "__main__":
    main()
