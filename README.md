# Industry RAG Assistants API

FastAPI backend for the AI assistant demos on roccocolasacco.com. Each assistant answers only from its own documents and cites the exact passages it used.

## How it works

1. **Search two ways at once.** Pinecone finds passages that mean the same thing as the question; BM25 keyword search finds passages that use the same words (names, numbers, rule references). The two result lists are merged.
2. **Split multi-part questions.** A quick model call turns a question like "when is the cutoff, and when is it due internally?" into separate searches, and each part gets its share of the results.
3. **Answer from the top passages only.** The 8 best passages (no more than 3 per file) are numbered and sent to the chat model, which must cite every fact like `[2]` or say it couldn't find the answer.
4. **Return the answer plus the cited passages** so the website can show sources.

Repeat questions are answered from memory instantly. Each visitor is limited to 20 questions per 10 minutes.

## Assistants

Defined in `assistants.json`; documents live in `knowledge/<assistant-id>/` (firm documents at the top level, official public documents in `public/`).

| id | Company (fictional) | Public documents |
|---|---|---|
| `law-firm` | Marlowe Kessler LLP (policies + a sample case file in `matters/`) | Federal Rules of Civil Procedure (Dec 1, 2025) |
| `property-management` | Larkspur Property Management | FTC: Using Consumer Reports: What Landlords Need to Know |
| `accounting` | Pell & Arden CPAs | IRS Publications 583 and 463 |
| `real-estate` | Bellwether Realty Group | CFPB: Your Home Loan Toolkit |
| `insurance` | Northgate Insurance Agency | FEMA: NFIP Summary of Coverage |

## Adding a new assistant (for example, a client)

1. Add an entry to `assistants.json`.
2. Put the documents (`.md`, `.txt`, `.pdf`) in `knowledge/<id>/`.
3. Run `python ingest.py <id>`. This embeds the documents, replaces that assistant's Pinecone namespace, and updates `chunks.json`.
4. Commit `chunks.json` and deploy.

## Local setup

Create `.env` (never commit it):

```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
```

```
pip install -r requirements.txt
python ingest.py
python main.py          # serves on http://localhost:8080
python eval.py          # runs the test questions in eval_questions.json
```

## Settings (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `CHAT_MODEL` | `gpt-4.1-mini` | Model that splits questions and writes answers (about $2 per 1,000 questions) |
| `REASONING_EFFORT` | unset | Only for reasoning models (for example `minimal`) |
| `PINECONE_INDEX` | `apex-solar-demo` | Pinecone index (one namespace per assistant) |
| `DEFAULT_ASSISTANT` | `law-firm` | Used when a request doesn't name one |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | `20` / `600` | Per-visitor limit |

## Keeping the free Render server awake

Render's free plan sleeps after 15 minutes without traffic. A free UptimeRobot HTTP monitor on `https://apex-solar-demo.onrender.com/` every 5 minutes keeps it awake.
