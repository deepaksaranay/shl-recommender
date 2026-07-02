# SHL Assessment Recommender

Conversational AI agent for helping hiring managers find the right SHL Individual Test Solutions.

## Setup & Run (3 steps)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env → add ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server starts using the included seed catalog (55 SHL assessments). To use the live SHL catalog:

```bash
# Requires playwright: pip install playwright && playwright install chromium
python scraper.py
# Then restart the server
```

## API

### GET /health
```json
{"status": "ok"}
```

### POST /chat
Send full conversation history on every call (stateless).

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I'm hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level are you targeting?"},
    {"role": "user", "content": "Mid-level, around 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are assessments suited for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Verify Inductive Reasoning", "url": "https://www.shl.com/...", "test_type": "A"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is `[]` when agent is clarifying or refusing.
`end_of_conversation` is `true` only when task is complete.
Max 8 messages per conversation (enforced).

## Test

```bash
python test_agent.py
```

## Deploy to Render (free tier)

1. Push this repo to GitHub
2. Go to render.com → New Web Service → connect repo
3. Environment variable: `ANTHROPIC_API_KEY` = your key
4. Render auto-detects `render.yaml` for config

## Project Structure

```
├── main.py             # FastAPI endpoints (/health, /chat)
├── agent.py            # LLM agent (claude-haiku, catalog-grounded)
├── catalog_search.py   # Self-contained BM25 retrieval + synonym expansion
├── scraper.py          # SHL catalog scraper (Playwright, JS rendering)
├── test_agent.py       # Behavioral test suite (6 tests)
├── approach.md         # Design document for submission
├── render.yaml         # One-click Render deployment
├── requirements.txt
├── Dockerfile
└── data/
    ├── seed_catalog.json   # 55 curated SHL Individual Test Solutions
    └── catalog.json        # Populated by scraper.py (git-ignored)
```
