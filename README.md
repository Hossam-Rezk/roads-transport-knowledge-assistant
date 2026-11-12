# Roads and Transport Knowledge Assistant

Evidence-grounded question answering over Dubai legislation in the **Roads and Transport**
primary category. FastAPI backend, LangGraph orchestration, LangChain agent with a tool and
middleware, hybrid retrieval (BM25 + embeddings + cross-encoder reranking), and a small
browser console for testing.

> The assistant summarises retrieved legislation text. It is not legal advice, the corpus is
> not exhaustive, and the Arabic original of every document prevails.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then fill in the Azure endpoint and key
uvicorn app.main:app --port 8000
```

Then open:

* `http://localhost:8000/` - chat page (live progress, cited answers, clarification, search options)
* `http://localhost:8000/docs` - generated OpenAPI documentation

On first use press **Build the index** on the chat page (or `POST /ingest`). The first search also
downloads the 34 MB reranker model once into `data/model_cache/`.

Run the tests (offline, no model needed): `pytest`

> **Windows note:** keep the project in a short path (for example `D:\rt-assistant`).
> `onnxruntime` fails to load when the virtual-environment path exceeds the Windows limit.

## Architecture

Layers only depend on the layers below them. `app/container.py` is the single place that
constructs and connects everything.

```
frontend/index.html          browser test console (static, served at /ui)
app/
  main.py                    FastAPI app, request-id HTTP middleware, error handlers, lifespan
  api/        routes.py      endpoints - validate, call a service, shape the response
              schemas.py     Pydantic request/response models
  services/   chat_service   per-thread locking, interrupt/resume, timeouts, error mapping
              search_service runs the retrieval subgraph on its own
  graph/      chat_graph     main StateGraph (planning, routing, interrupt, answer, citations)
              retrieval_graph reusable subgraph (parallel search, fuse, rerank)
              query_planner  LLM -> Pydantic QueryPlan
              answer_agent   create_agent + tool + middleware -> Pydantic AnswerDraft
              tools.py       get_document_excerpt (the model-selected tool)
              middleware.py  LangChain agent middleware (timing, controlled tool failure)
              state.py       typed state, with notes on what persists and what resets
              prompts.py     all prompt text
  retrieval/  retrievers     LexicalRetriever (FTS5/BM25), SemanticRetriever (Chroma)
              filters        SearchFilters -> SQL WHERE / Chroma where
              fusion         reciprocal rank fusion
              reranker       FlashRank cross-encoder
  ingestion/  manifest, pdf_parser, chunker, pipeline
  storage/    chunk_store (SQLite + FTS5), vector_store (Chroma)
  core/       models, errors, logging_setup
  config.py   every tunable setting      llm.py   chat + embedding model factories
scripts/      fetch_corpus.py (acquire corpus), export_graph.py (Mermaid diagram)
tests/        unit tests with mocked model responses
data/         manifest.json, evidence/, raw/pdf, raw/html, index/ (generated)
```

## Corpus

* Source: Dubai Legislation Portal, Legislation Search Engine, filter **Primary = "Roads and Transport"**.
* `scripts/fetch_corpus.py` applies the filter, pages through the results, saves the raw
  filtered result pages to `data/evidence/`, writes `data/manifest.json` and downloads each
  document as PDF and HTML. Re-running skips existing files.
* The filter returned **45 documents and all 45 are indexed** (856 chunks). The brief
  suggests 15-25; the whole category was used so that no selection bias is introduced and
  amendment families stay together.
* Ingestion only accepts manifest entries whose category is "Roads and Transport". Any other
  entry, and any PDF not listed in the manifest, is excluded and reported.

## Ingestion and chunking

1. **Parse** each PDF page with PyMuPDF (all 45 PDFs are text-based).
2. **Clean** the running header (title + "Page X of N"), the copyright/translation footer and
   separator lines.
3. **Segment** at the document's own boundaries: preamble, one segment per `Article (n)`
   (the line above the marker is the article title), one per `Schedule (n)`.
4. **Chunk**: a segment that fits `CHUNK_SIZE` (1200 characters) is one chunk; longer ones are
   split on line/clause boundaries with `CHUNK_OVERLAP` (150) characters repeated.
5. **Index text** = document title + article heading + chunk text, so a bare clause keeps its context.

Chunk ids are `<document_id>-<segment index>-<part index>` (for example `dlp-1525-011-1`),
which is deterministic. A document is skipped when its file hash is unchanged, and replaced
as a whole when it changed, so re-ingestion never creates duplicates.

Metadata per chunk: chunk id, document id, title, source URL, primary category, locator
(`pdf_page`, e.g. `6` or `6-7`), article number and title, plus type, number, year, authority
and status from the manifest (null when the portal has no value).

## Retrieval

| Stage | Implementation |
|---|---|
| Lexical | SQLite FTS5, BM25 ranking |
| Semantic | Chroma (cosine) with Azure `text-embedding-3-small` |
| Hybrid | both of the above as **parallel LangGraph nodes** |
| Fusion | reciprocal rank fusion (`RRF_K` = 60), deduplicated by chunk id. Uses rank positions only; raw BM25 and cosine values are never added together |
| Reranking | FlashRank cross-encoder `ms-marco-MiniLM-L-12-v2`, running locally on CPU |
| Filters | `year`, `legislation_type`, `status`, plus the always-on category restriction. Applied inside each store's query, before ranking |

Each branch returns up to `CANDIDATES_PER_BRANCH` (20) chunks; the reranker returns the final
`top_k` (1-20, default 5). `/search` labels every score with its type; scores of different
types are not comparable and are not confidence values. `rerank: false` returns the
pre-rerank order so the effect of reranking can be inspected.

## Graph

`python scripts/export_graph.py` writes the diagram of the compiled graph to `docs/graph.mmd`.

```
start_turn -> plan_query -+-> scope_response -> END                      (out of scope)
                          +-> ask_clarification -> plan_query            (interrupt, max 2)
                          +-> retrieve [subgraph] -+-> insufficient_evidence_response -> END
                                                   +-> complete_split_articles
                                                       -> generate_answer -> validate_citations -> END

retrieve:  START -+-> lexical_search --+-> fuse_candidates -> rerank_candidates -> END
                  +-> semantic_search -+
```

* **State.** `messages` accumulates across turns (thread memory). Every other field belongs to
  the current question and is reset by `start_turn`.
* **Parallel writes.** The two search nodes write separate keys (`lexical_candidates`,
  `semantic_candidates`). `fuse_candidates` runs once, in the step after both have finished.
* **Clarification.** `ask_clarification` calls `interrupt()`; `/chat/resume` continues the same
  run with `Command(resume=...)`. Nothing with side effects runs before the interrupt, because
  the node re-executes from its first line on resume. At most two attempts per question.
* **Persistence.** `AsyncSqliteSaver` in `data/index/checkpoints.sqlite`, keyed by `thread_id`.
  This is LangGraph's per-thread run state (it survives restarts); it is not a long-term
  cross-thread memory store.
* **Split articles.** If retrieval selected only one part of a long article,
  `complete_split_articles` adds the other parts so lists are not answered half-way.
* **Citation validation.** A citation is accepted only if the chunk was shown to the model
  (final evidence or a tool result) and exists in the index. An answer with no valid citation
  becomes `insufficient_evidence`.

## Tool calling and middleware

The answer step is a LangChain `create_agent` embedded as a graph node.

* **Tool** `get_document_excerpt(document_id, locator)` - the model decides when to call it
  (for example "What does Article 14 of ... say?" when that article is not in the evidence).
  Arguments are validated by Pydantic and lookups are restricted to the indexed corpus.
* **Middleware** (LangChain agent middleware, not HTTP middleware) wraps each model call and
  tool call inside the agent loop:
  `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ModelRetryMiddleware`, and the custom
  `ExecutionLoggingMiddleware` (timing logs; a failing tool call is returned to the model as an
  error message instead of crashing the run).
* The FastAPI HTTP middleware in `main.py` is separate: it assigns a request id and logs each
  HTTP request.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness and index availability |
| `POST /ingest` | ingest the corpus; returns indexed / skipped / failed / excluded counts and errors |
| `POST /search` | retrieval only, with mode, filters, `top_k`, `rerank` |
| `POST /chat` | ask a question; returns `answered`, `needs_clarification`, `insufficient_evidence` or `out_of_scope` |
| `POST /chat/resume` | answer a pending clarification |
| `POST /chat/stream`, `POST /chat/resume/stream` | same as the two above, streamed as Server-Sent Events (used by the chat page) |
| `GET /threads/{thread_id}` | inspect a thread's saved state (extra, for demos) |

```bash
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d "{}"
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
     -d "{\"query\": \"Who is exempt from the traffic toll?\", \"mode\": \"hybrid\", \"filters\": {\"legislation_type\": \"Law\"}, \"top_k\": 5}"
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d "{\"thread_id\": \"demo-1\", \"message\": \"What does the law say about licensing autonomous vehicles?\"}"
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d "{\"thread_id\": \"demo-2\", \"message\": \"What is the fee?\"}"
curl -X POST localhost:8000/chat/resume -H "Content-Type: application/json" \
     -d "{\"thread_id\": \"demo-2\", \"clarification\": \"The Dubai airports departure fee\"}"
```

**Streaming.** The stream sends `stage` events as each graph step starts ("Searching the
legislation", "Checking every citation", ...), a `sources` event with the documents selected
as evidence, then one `final` event containing the normal `ChatResponse` (or an `error`
event). The answer text itself is sent only after citation validation, so unvalidated model
output never reaches the user; the chat page then reveals it with a typing effect.

Limits: messages and queries up to 2000 characters; `top_k` 1-20; one run per thread at a
time (a concurrent request gets `409 thread_busy`). Controlled errors use the shape
`{"error": {"code", "message"}, "request_id"}`: `index_unavailable` (503),
`clarification_pending` / `invalid_resume` / `thread_busy` (409), `model_failure` (502),
`request_timeout` (504). Invalid input returns FastAPI's 422 validation error.

## Logging

JSON lines on the console and in `logs/app.log`. Every line carries `request_id` and
`thread_id`. Events include `node_start` / `node_end` (with candidate counts and duration),
`query_planned`, `middleware_model_call`, `middleware_tool_call`, `citations_validated` and
`http_request`. Credentials and full document text are never logged.

## Model configuration

| Setting | Value |
|---|---|
| Chat model | Azure AI Foundry deployment `gpt-4.1-mini`, temperature 0 |
| Embeddings | Azure AI Foundry deployment `text-embedding-3-small` (1536 dimensions) |
| Endpoint | OpenAI-compatible `/openai/v1`, used through `langchain-openai` |
| Reranker | `ms-marco-MiniLM-L-12-v2` via FlashRank (local) |

Versions are pinned in `requirements.txt`; `requirements.lock.txt` is the full freeze of the
environment the project was built and tested with (Python 3.13).

## Known limitations

* Tables in schedules are flattened to text by PDF extraction; the cell order is kept but the
  grid is lost. A few definition tables come out with the definition before the term.
* Only PDFs are parsed. The HTML renditions are downloaded but an HTML fallback parser is not
  implemented (it was not needed: all 45 PDFs contain text).
* Amending resolutions quote the articles they replace, so an article number can occur more
  than once inside one document.
* The reranker is a general-purpose English model, not trained on legal text.
* The query planner is an LLM, so scope and ambiguity decisions can occasionally vary.
* The per-thread lock is in-process; running several server processes would need a shared lock.
* Not yet done from the brief: the completed acceptance checklist and a portal screenshot of
  the applied filter.

## Evaluation

`evaluation/` holds a 12-question test set with reference answers, a retrieval check
(expected document in the top 5 per search mode) and RAGAS scoring. Latest run: 12/12
expected documents found in every mode; faithfulness 0.99, context recall 1.00, context
precision 0.95, answer relevancy 0.85. See `evaluation/README.md` for
how to run it and how to read the scores, including their limitations.
