# main.py - FastAPI RAG API: one assistant per industry, each answering only from its own documents.
#
# Search is hybrid: Pinecone finds passages that MEAN the same thing as the question,
# and BM25 keyword search finds passages that use the same WORDS (names, product
# codes, "Concur", rule numbers). The two ranked lists are merged, the best passages
# are numbered, and the chat model answers from them with [n] citations.
import json
import os
import re
import time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from settings import (ALLOWED_ORIGIN_REGEX, ALLOWED_ORIGINS, CHAT_MODEL, DEFAULT_ASSISTANT,
                      EMBED_MODEL, PINECONE_INDEX, RATE_LIMIT_REQUESTS,
                      RATE_LIMIT_WINDOW_SECONDS)

ROOT = Path(__file__).parent
ASSISTANTS = json.loads((ROOT / "assistants.json").read_text(encoding="utf-8"))

CANDIDATES = 20      # passages pulled from each search method
CONTEXT_PASSAGES = 8  # passages given to the model
MAX_PER_DOCUMENT = 3  # passages allowed from any one file
RRF_K = 60           # standard reciprocal-rank-fusion constant

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it
its me my of on or our should so that the their them there this to us was we what when
where which who why will with you your""".split())

app = FastAPI(title="Industry RAG Assistants API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_index = None
chunks_by_assistant: dict = {}
chunk_lookup: dict = {}
bm25_by_assistant: dict = {}
answer_cache: OrderedDict = OrderedDict()
requests_by_ip = defaultdict(deque)


def tokenize(text: str) -> List[str]:
    tokens = (t.strip(".") for t in re.findall(r"[a-z0-9$%.]+", text.lower().replace(",", "")))
    return [t for t in tokens if t and t not in STOPWORDS]


@app.on_event("startup")
def startup():
    global pinecone_index
    all_chunks = json.loads((ROOT / "chunks.json").read_text(encoding="utf-8"))
    for assistant_id, chunks in all_chunks.items():
        chunks_by_assistant[assistant_id] = chunks
        for c in chunks:
            chunk_lookup[c["id"]] = c
        bm25_by_assistant[assistant_id] = BM25Okapi(
            [tokenize(f"{c['title']} {c['location']} {c['text']}") for c in chunks])
    print(f"Loaded {len(chunk_lookup)} chunks for {len(all_chunks)} assistants")

    pinecone_index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(PINECONE_INDEX)
    stats = pinecone_index.describe_index_stats()  # also warms up the connection
    print(f"Pinecone '{PINECONE_INDEX}' has {stats.total_vector_count} vectors")


# === Models ===
class Query(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    assistant: Optional[str] = None


class Source(BaseModel):
    n: int
    filename: str
    title: str
    location: str
    text: str
    public: bool


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    assistant: str
    cached: bool
    ms: int


# === Retrieval ===
SPLIT_PROMPT = """Turn the user's question into 1 to 3 short search queries for a document search, one per distinct thing being asked.
Keep names, numbers, and key terms. When part of the question is about the organization's general rules or policies, write that query WITHOUT the case, client, or person's name.
If it asks a single thing, return just one query. Reply as JSON: {"queries": ["..."]}

Example: "When is the Smith expert deadline and when do we need it done internally?"
-> {"queries": ["Smith case expert disclosure deadline", "firm policy internal deadline before court deadlines"]}"""


def split_question(question: str) -> List[str]:
    """Break multi-part questions into separate searches. Falls back to the question itself."""
    try:
        kwargs = {"temperature": 0} if CHAT_MODEL.startswith("gpt-4") else {}
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": SPLIT_PROMPT}, {"role": "user", "content": question}],
            response_format={"type": "json_object"},
            max_completion_tokens=150,
            **kwargs,
        )
        queries = json.loads(response.choices[0].message.content or "{}").get("queries", [])
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()][:3]
    except Exception as e:
        print(f"Question split failed, searching the original question only: {e!r}")
        queries = []
    return [question] + [q for q in queries if q.lower() != question.lower()]


def keyword_search(assistant_id: str, query: str) -> List[str]:
    bm25 = bm25_by_assistant[assistant_id]
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
    chunks = chunks_by_assistant[assistant_id]
    return [chunks[i]["id"] for i in ranked if scores[i] > 0]


def meaning_search(assistant_id: str, vector: List[float]) -> List[str]:
    result = pinecone_index.query(vector=vector, top_k=CANDIDATES, namespace=assistant_id)
    return [m.id for m in result.matches if m.id in chunk_lookup]


def fuse(ranked_lists: List[List[str]]) -> List[str]:
    """Reciprocal rank fusion: merge several ranked lists into one."""
    scores = defaultdict(float)
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] += 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(assistant_id: str, question: str) -> List[dict]:
    queries = split_question(question)
    vectors = [d.embedding for d in openai_client.embeddings.create(model=EMBED_MODEL, input=queries).data]
    with ThreadPoolExecutor(max_workers=len(vectors)) as pool:
        meaning_lists = list(pool.map(lambda v: meaning_search(assistant_id, v), vectors))

    # One merged (meaning + keyword) list per query, then take turns between the
    # queries, so each part of a multi-part question gets passages, not just the part
    # whose words appear most often.
    per_query = [fuse([meaning_lists[i], keyword_search(assistant_id, q)]) for i, q in enumerate(queries)]

    # No more than MAX_PER_DOCUMENT from any one file, so one long document can't
    # crowd out a short policy that answers part of the question.
    best, chosen, per_doc = [], set(), defaultdict(int)
    positions = [0] * len(per_query)
    while len(best) < CONTEXT_PASSAGES and any(p < len(lst) for p, lst in zip(positions, per_query)):
        for i, ranked in enumerate(per_query):
            while positions[i] < len(ranked):
                chunk_id = ranked[positions[i]]
                positions[i] += 1
                filename = chunk_lookup[chunk_id]["filename"]
                if chunk_id not in chosen and per_doc[filename] < MAX_PER_DOCUMENT:
                    best.append(chunk_lookup[chunk_id])
                    chosen.add(chunk_id)
                    per_doc[filename] += 1
                    break
            if len(best) == CONTEXT_PASSAGES:
                break
    return best


# === Generation ===
NOT_FOUND = "I couldn't find that in the documents"


def system_prompt(assistant_id: str) -> str:
    a = ASSISTANTS[assistant_id]
    return f"""You are the internal knowledge assistant for {a['company']}, {a['description']}. You help {a['audience']}.

Answer ONLY from the numbered sources provided with each question. Rules:
- Every sentence that states a fact must end with a citation like [2], using the source numbers given. If you cannot cite it, do not say it. Never cite a number that was not given.
- If the sources do not answer the question, start with exactly: "{NOT_FOUND}." Then, if the sources contain closely related facts, share them with citations (for example, who holds a related role). Otherwise add one short sentence on who or what to check. Never guess or use outside knowledge.
- When a question involves a deadline or date, give the actual date from the sources, and mention any firm policy that changes when work is internally due.
- Firm documents are {a['company']}'s own policies. Public documents are official government publications. If both apply, say which is firm policy and which is the official rule.
- Be direct and specific: lead with the answer (numbers, dates, names), in 1-4 short sentences or a short bulleted list. No preamble."""


def build_user_message(question: str, passages: List[dict]) -> str:
    blocks = []
    for n, c in enumerate(passages, start=1):
        kind = "Public document" if c["public"] else "Firm document"
        blocks.append(f"[{n}] {kind}: {c['title']} ({c['location']})\n{c['text']}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def ask_model(assistant_id: str, question: str, passages: List[dict]) -> str:
    kwargs = {}
    if CHAT_MODEL.startswith("gpt-4"):
        kwargs["temperature"] = 0
    if os.getenv("REASONING_EFFORT"):
        kwargs["reasoning_effort"] = os.getenv("REASONING_EFFORT")
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt(assistant_id)},
            {"role": "user", "content": build_user_message(question, passages)},
        ],
        max_completion_tokens=int(os.getenv("MAX_ANSWER_TOKENS", "600")),
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip()


def cited_sources(answer: str, passages: List[dict]) -> List[Source]:
    cited = []
    for match in re.findall(r"\[(\d+)\]", answer):
        n = int(match)
        if 1 <= n <= len(passages) and n not in cited:
            cited.append(n)
    return [Source(n=n, filename=passages[n - 1]["filename"], title=passages[n - 1]["title"],
                   location=passages[n - 1]["location"], text=passages[n - 1]["text"],
                   public=passages[n - 1]["public"]) for n in sorted(cited)]


# === Abuse protection ===
def check_rate_limit(request: Request):
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.time()
    window = requests_by_ip[ip]
    while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many questions in a short time. Please wait a few minutes.")
    window.append(now)


# === Endpoints ===
@app.get("/")
def health():
    """Health check. UptimeRobot pings this to keep the free Render server awake."""
    return {"status": "ok", "assistants": list(ASSISTANTS), "chunks": len(chunk_lookup)}


@app.post("/query/", response_model=QueryResponse)
def query(q: Query, request: Request):
    started = time.perf_counter()
    assistant_id = q.assistant or DEFAULT_ASSISTANT
    if assistant_id not in chunks_by_assistant:
        raise HTTPException(status_code=404, detail=f"Unknown assistant '{assistant_id}'")

    question = " ".join(q.question.split())
    cache_key = (assistant_id, question.lower().rstrip("?. "))
    if cache_key in answer_cache:
        answer_cache.move_to_end(cache_key)
        cached = answer_cache[cache_key]
        return QueryResponse(**cached, assistant=assistant_id, cached=True,
                             ms=int((time.perf_counter() - started) * 1000))

    check_rate_limit(request)
    try:
        passages = hybrid_search(assistant_id, question)
        answer = ask_model(assistant_id, question, passages) if passages else f"{NOT_FOUND}."
    except Exception as e:
        print(f"Query failed: {e!r}")
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable. Please try again.")

    sources = cited_sources(answer, passages)
    result = {"answer": answer, "sources": sources}
    answer_cache[cache_key] = result
    if len(answer_cache) > 500:
        answer_cache.popitem(last=False)
    return QueryResponse(**result, assistant=assistant_id, cached=False,
                         ms=int((time.perf_counter() - started) * 1000))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
