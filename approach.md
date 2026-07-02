# SHL Assessment Recommender — Approach Document

## Design Overview

The system is a stateless FastAPI service with two endpoints (`GET /health`, `POST /chat`). Each call receives the full conversation history, retrieves catalog-relevant context via BM25, and makes a single LLM call (claude-haiku) that returns structured JSON.

### Architecture

```
User → POST /chat → CatalogSearch (BM25) → LLM (claude-haiku) → Validated JSON → Response
```

**No per-conversation server state.** The full history is sent by the caller on every turn, keeping deployment simple (Render/Fly free tier, no persistent storage needed).

---

## Retrieval Setup

**BM25 (rank-bm25)** over the SHL Individual Test Solutions catalog.

**Why BM25 over embeddings?**
- Zero warmup latency (no model loading) — critical for the 30s timeout
- Deterministic results, easier to debug
- Sufficient for a catalog of ~40–100 items with well-written descriptions
- No GPU or external vector store needed

**Query expansion:** A synonym map boosts recall for common mismatches (`developer → programming, Java, Python`; `manager → leadership, team`). User messages from all turns are concatenated before search so later context refines earlier retrieval.

**Catalog loading:** On startup, prefers `data/catalog.json` (populated by `scraper.py`) and falls back to `data/seed_catalog.json` (curated manually). The scraper fetches SHL's product catalog page, paginates through all Individual Test Solutions, and optionally scrapes detail pages for descriptions and job levels.

---

## Agent / Prompt Design

### System prompt strategy
The system prompt embeds:
1. A behavioral decision tree (clarify → recommend → refine → compare → refuse)
2. Strict rules about catalog-only URLs
3. The BM25-retrieved catalog slice (top 15-18 items), formatted compactly
4. A required JSON output schema

**Catalog injection:** Only relevant items are injected (not the full catalog) to keep the context short and the LLM focused. A small augmentation always adds 2 personality + 1 motivation option, since users often ask for these as refinements.

### Behavioral rules
- **Clarify**: Agent asks ONE question per turn. Never recommends on turn 1 for vague queries.
- **Recommend**: Triggers when role + at minimum one qualifier (seniority or competency focus) is known.
- **Refine**: History is included in every call, so the LLM sees the prior shortlist and updates it.
- **Compare**: Catalog descriptions are in context; LLM draws from those rather than priors.
- **Refuse**: System prompt explicitly lists out-of-scope categories.

### Model choice
**claude-haiku-4-5** — fastest Anthropic model, consistently responds in < 5s. Leaves headroom within the 30s timeout for BM25 retrieval (~1ms) and network overhead.

### Anti-hallucination
After parsing the LLM's JSON, every recommendation URL is checked against the catalog's known URL set. Hallucinated URLs are dropped. A secondary name-match lookup attempts recovery (LLM gave right name but wrong URL slug).

---

## Evaluation Approach

### Hard evals
Schema compliance is enforced by Pydantic at the API layer — malformed responses raise 422 before reaching the evaluator. Turn cap (8) is validated on `POST /chat`.

### Recall@10
Tested against the 10 public conversation traces. Main levers:
- BM25 with query expansion improves recall for synonym-heavy queries
- Injecting personality/motivation items regardless of query ensures they appear as options
- Multi-turn context aggregation (all user messages used for search) helps when key terms appear late

### Behavior probes
Manual tests cover: vague-query clarification (turn 1), off-topic refusal, prompt injection, mid-conversation refinement, comparison grounding. These are automated in `test_agent.py`.

---

## What Didn't Work

- **Embeddings + FAISS**: Slow cold start on free-tier hosting; caused timeout failures before switching to BM25.
- **Tool-use (function calling)**: Added latency with a second LLM call for retrieval decisions. Single-pass prompt with pre-retrieved catalog context is faster and more reliable.
- **Asking multiple clarifying questions**: Users found multi-question turns unnatural. Reduced to one question per turn.

---

## Tools Used

- Claude (Anthropic) — assisted with initial prompt drafts and test case generation.
- All design decisions, architecture, and implementation independently verified and authored.

## Stack
FastAPI · Anthropic (claude-haiku) · rank-bm25 · BeautifulSoup4 · Pydantic · Uvicorn
Deployment: Render (free tier) or Fly.io
