# Evaluation

Two checks, run against the real pipeline (Azure models, the full 45-document index):

1. **Retrieval check** (brief AT-12): for each question, is an expected source document in
   the final top 5, in lexical, semantic and hybrid mode?
2. **RAGAS** answer-quality scoring with an LLM judge.

## How to run

The application environment collects answers; RAGAS runs in its own environment because it
pins older versions of `langchain-community` and `openai` than the application uses.

```bash
python evaluation/collect_answers.py
```

```bash
python -m venv .venv-eval
```

```bash
.venv-eval\Scripts\pip install -r evaluation/requirements-eval.txt
```

```bash
.venv-eval\Scripts\python evaluation/score_with_ragas.py
```

| File | Content |
|---|---|
| `testset.json` | 12 questions, each with a reference answer and the source chunk ids it was written from |
| `collect_answers.py` | runs each question through the chat graph (hybrid, top 5) and the three search modes |
| `score_with_ragas.py` | RAGAS scoring, judge = `gpt-4.1-mini` |
| `results/samples.json` | question, answer, citations, the exact evidence given to the model |
| `results/retrieval_hits.md` | top-5 document check per search mode |
| `results/ragas_scores.csv`, `results/ragas_report.md` | per-question scores |

## How the test set was built

Each reference answer was written by reading the cited chunk in the index, not by the
system under test. The questions cover 9 documents and 6 topics: traffic toll (law,
fees, fines), airport departure fee, autonomous vehicles, parking peak hours, water taxis,
drones and driving instruction.

## Results (run of 2026-11-12)

### Retrieval: expected document in the final top 5

| Mode | Hit rate | Rank 1 |
|---|---|---|
| Lexical | 12/12 | 11/12 |
| Semantic | 12/12 | 11/12 |
| Hybrid | 12/12 | 11/12 |

The one rank-2 case (q01) ranked the implementing bylaw of the same toll law first, which
is a related, reasonable document.

### RAGAS (12 answered questions, 0 abstentions)

| Metric | Mean | What it measures |
|---|---|---|
| Faithfulness | **0.99** | answer claims supported by the retrieved evidence |
| Context recall | **1.00** | evidence contains what the reference answer needs |
| Context precision | **0.95** | relevant evidence ranked above irrelevant evidence |
| Answer relevancy | **0.85** | answer addresses the question asked |

Per-question scores: `results/ragas_report.md`.

## Reading the results

- **Grounding is the strongest result.** Faithfulness 0.99 and context recall 1.00 mean the
  answers stay within the evidence and the evidence contains what is needed. This is the
  property the design (citation validation, evidence gate) was built for.
- **Answer relevancy 0.85** reflects extra detail: the references are one or two sentences,
  while answers that add related, correct and cited provisions (for example q11 lists five
  separate drone permits) are judged slightly less focused.

## Limitations of this evaluation

- **Small and optimistic.** 12 questions, written by someone who knew which chunk holds the
  answer, so the wording tends to match the source. The 12/12 retrieval hit rate should not
  be read as general recall; it also does not separate the three search modes.
- **No negative cases scored.** Out-of-scope, unsupported and ambiguous questions are not
  in this set; RAGAS has nothing to score for an abstention. They are covered by the unit
  tests and the manual checks instead.
- **LLM judge.** The judge is the same model family as the system, and scores can vary
  between runs. Treat the numbers as indicators.
- **Next steps:** add paraphrased and multi-document questions that favour one search mode,
  add negative cases with an expected status, and repeat runs to report a range.
