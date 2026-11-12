"""Evaluation step 1 of 2: run every test question through the real pipeline.

Runs in the APPLICATION environment (the same one that runs the API):

    python evaluation/collect_answers.py

For each question in evaluation/testset.json it records
  * the answer produced by the chat graph (hybrid mode) and its citations
  * the exact evidence chunks the answer model was given (the "contexts")
  * for each search mode, whether an expected document is in the final top 5

Output: evaluation/results/samples.json  (input for score_with_ragas.py)
        evaluation/results/retrieval_hits.md  (the top-5 document check)
"""
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.container import Container, build_container  # noqa: E402
from app.core.models import SearchFilters, SearchMode  # noqa: E402
from app.graph.prompts import DISCLAIMER  # noqa: E402

EVALUATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALUATION_DIR / "results"
TOP_K = 5
CITATION_MARKER = re.compile(r"\s*\[[A-Za-z]+-\d+-\d+-\d+(?:[,;\s]+[A-Za-z]+-\d+-\d+-\d+)*\]")


def clean_answer(answer_text: str) -> str:
    """Remove the fixed disclaimer and the [chunk-id] markers so that the
    judge scores only the statements the model made."""
    without_disclaimer = answer_text.replace(DISCLAIMER, "")
    return CITATION_MARKER.sub("", without_disclaimer).strip()


async def expected_document_rank(container: Container, question: dict, mode: SearchMode) -> int | None:
    """Position (1-5) of the first expected document in the final results, or None."""
    outcome = await container.search_service.search(
        question["question"], mode, SearchFilters(), TOP_K)
    for position, retrieved in enumerate(outcome.results, start=1):
        if retrieved.chunk.document_id in question["expected_document_ids"]:
            return position
    return None


async def answer_question(container: Container, question: dict) -> dict:
    """Run the full chat graph on a fresh thread and collect what it produced."""
    state = await container.chat_graph.ainvoke(
        {"user_message": question["question"], "search_mode": SearchMode.HYBRID.value,
         "request_filters": {}, "top_k": TOP_K},
        {"configurable": {"thread_id": f"evaluation-{question['id']}"}})

    if "__interrupt__" in state:        # the planner asked for a clarification
        status, answer_text, citations = "needs_clarification", "", []
    else:
        answer = state["answer"]
        status, answer_text, citations = answer["status"], answer["answer"], answer["citations"]

    evidence = state.get("final_evidence", [])
    return {
        "id": question["id"],
        "question": question["question"],
        "reference": question["reference"],
        "expected_document_ids": question["expected_document_ids"],
        "status": status,
        "answer": clean_answer(answer_text),
        "cited_chunk_ids": [citation["chunk_id"] for citation in citations],
        "context_chunk_ids": [item["chunk"]["chunk_id"] for item in evidence],
        "contexts": [item["chunk"]["search_text"] for item in evidence],
    }


def write_retrieval_report(samples: list[dict]) -> None:
    modes = [mode.value for mode in SearchMode]
    lines = ["# Retrieval check: is an expected document in the final top 5?", "",
             "Each cell is the rank (1-5) of the first expected document, or `miss`.", "",
             "| Question | Expected | " + " | ".join(modes) + " |",
             "|---|---|" + "---|" * len(modes)]
    for sample in samples:
        ranks = [str(sample["expected_rank"][mode] or "miss") for mode in modes]
        lines.append(f"| {sample['id']} {sample['question']} | "
                     f"{', '.join(sample['expected_document_ids'])} | " + " | ".join(ranks) + " |")
    hits = [sum(1 for sample in samples if sample["expected_rank"][mode]) for mode in modes]
    lines += ["", "| Hit rate | | " + " | ".join(f"{hit}/{len(samples)}" for hit in hits) + " |"]
    (RESULTS_DIR / "retrieval_hits.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    logging.disable(logging.CRITICAL)                  # keep the console readable
    questions = json.loads((EVALUATION_DIR / "testset.json").read_text(encoding="utf-8"))["questions"]
    container = build_container(get_settings(), InMemorySaver())
    if container.chunk_store.count_chunks() == 0:
        sys.exit("The index is empty. Start the API and run POST /ingest first.")

    samples = []
    for question in questions:
        sample = await answer_question(container, question)
        sample["expected_rank"] = {
            mode.value: await expected_document_rank(container, question, mode)
            for mode in SearchMode}
        samples.append(sample)
        print(f"{sample['id']}  {sample['status']:<22} expected-document rank: {sample['expected_rank']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    write_retrieval_report(samples)
    print(f"\nwrote {RESULTS_DIR / 'samples.json'} and retrieval_hits.md")


if __name__ == "__main__":
    asyncio.run(main())
