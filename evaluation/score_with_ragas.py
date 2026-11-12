"""Evaluation step 2 of 2: score the collected answers with RAGAS.

Runs in a SEPARATE environment, because RAGAS pins different versions of some
packages than the application does:

    python -m venv .venv-eval
    .venv-eval\\Scripts\\pip install -r evaluation/requirements-eval.txt
    .venv-eval\\Scripts\\python evaluation/score_with_ragas.py

Input:  evaluation/results/samples.json   (from collect_answers.py)
Output: evaluation/results/ragas_scores.csv and ragas_report.md

Metrics (all between 0 and 1, higher is better)
  faithfulness         are the answer's statements supported by the retrieved contexts?
  answer_relevancy     does the answer address the question that was asked?
  context_precision    are the relevant contexts ranked above the irrelevant ones?
  context_recall       do the contexts contain what the reference answer needs?

The judge is an LLM, so scores vary slightly between runs and are indicators,
not exact measurements.
"""
import json
import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"

# RAGAS column name -> the shorter name used in the report.
METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


def load_env_file(env_path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment (no extra dependency)."""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def build_judge():
    """The judge uses the same Azure deployments as the application."""
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    chat_model = ChatOpenAI(model=os.environ.get("CHAT_DEPLOYMENT", "gpt-4.1-mini"),
                            base_url=endpoint, api_key=api_key, temperature=0)
    embeddings = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        base_url=endpoint, api_key=api_key, check_embedding_ctx_length=False)
    return LangchainLLMWrapper(chat_model), LangchainEmbeddingsWrapper(embeddings)


def write_report(scored_rows: list[dict], not_scored: list[dict], judge_model: str) -> None:
    metric_columns = list(METRIC_NAMES.values())
    lines = ["# RAGAS evaluation report", "",
             f"Judge model: `{judge_model}`. Scores are 0-1, higher is better.", "",
             "| Question | " + " | ".join(metric_columns) + " |",
             "|---|" + "---|" * len(metric_columns)]
    for row in scored_rows:
        cells = [f"{row[column]:.2f}" if row[column] is not None else "n/a"
                 for column in metric_columns]
        lines.append(f"| {row['id']} {row['question']} | " + " | ".join(cells) + " |")

    means = []
    for column in metric_columns:
        values = [row[column] for row in scored_rows if row[column] is not None]
        means.append(f"**{sum(values) / len(values):.2f}**" if values else "n/a")
    lines.append("| **Mean** | " + " | ".join(means) + " |")

    if not_scored:
        lines += ["", "## Not scored", "",
                  "These questions did not produce an answer, so there is nothing to score.", ""]
        lines += [f"- {row['id']} {row['question']} -> `{row['status']}`" for row in not_scored]
    (RESULTS_DIR / "ragas_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    samples_path = RESULTS_DIR / "samples.json"
    if not samples_path.exists():
        sys.exit("Run evaluation/collect_answers.py first.")
    load_env_file(PROJECT_DIR / ".env")
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    # Abstentions and clarification requests contain no claims to judge.
    answered = [sample for sample in samples if sample["status"] == "answered"]
    not_scored = [sample for sample in samples if sample["status"] != "answered"]

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(user_input=sample["question"], response=sample["answer"],
                         retrieved_contexts=sample["contexts"], reference=sample["reference"])
        for sample in answered])
    judge_llm, judge_embeddings = build_judge()
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                 LLMContextRecall()],
        llm=judge_llm, embeddings=judge_embeddings)

    scores = result.to_pandas()
    scored_rows = []
    for sample, (_, score_row) in zip(answered, scores.iterrows()):
        row = {"id": sample["id"], "question": sample["question"]}
        for ragas_column, report_column in METRIC_NAMES.items():
            value = score_row.get(ragas_column)
            row[report_column] = None if value is None or value != value else float(value)  # NaN check
        scored_rows.append(row)

    scores.insert(0, "id", [sample["id"] for sample in answered])
    scores.to_csv(RESULTS_DIR / "ragas_scores.csv", index=False, encoding="utf-8")
    write_report(scored_rows, not_scored, os.environ.get("CHAT_DEPLOYMENT", "gpt-4.1-mini"))
    print((RESULTS_DIR / "ragas_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
