# ingest.py - turn the documents in knowledge/<assistant>/ into searchable chunks.
#
# Usage:
#   python ingest.py                 # rebuild every assistant
#   python ingest.py law-firm        # rebuild one assistant
#
# For each assistant this:
#   1. reads its Markdown (.md/.txt) and PDF files (firm docs at the top level,
#      public reference docs in public/),
#   2. splits them into ~300-word chunks that keep their document title,
#      section heading and page number,
#   3. embeds the chunks with OpenAI (in batches),
#   4. replaces that assistant's Pinecone namespace with the new vectors, and
#   5. writes chunks.json, which the API loads for keyword search and citations.
import json
import re
import sys
from pathlib import Path

import pymupdf
from openai import OpenAI
from pinecone import Pinecone

from settings import EMBED_MODEL, PINECONE_INDEX

ROOT = Path(__file__).parent
KNOWLEDGE = ROOT / "knowledge"
CHUNKS_FILE = ROOT / "chunks.json"

CHUNK_WORDS = 300
OVERLAP_WORDS = 50


def clean(text: str) -> str:
    """Normalise quotes, re-join words hyphenated across PDF line breaks, squash spaces."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("�", "'").replace(" ", " ")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_words(words, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Yield (start, end) word windows with overlap."""
    if len(words) <= size:
        yield 0, len(words)
        return
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield start, end
        if end == len(words):
            break
        start = end - overlap


def title_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ")


def chunk_markdown(path: Path):
    """Split a Markdown file by headings, then by size. Location = section heading."""
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", raw, re.MULTILINE)
    title = clean(title_match.group(1)) if title_match else title_from_filename(path)

    sections = []
    heading = "Overview"
    buf = []
    for line in raw.splitlines():
        m = re.match(r"^(#{1,3}) (.+)$", line)
        if m:
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading = clean(m.group(2)) if len(m.group(1)) > 1 else "Overview"
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf)))

    for heading, body in sections:
        # Keep line breaks (as their own tokens) so lists and tables stay readable when quoted.
        lines = [line.strip() for line in clean(body).splitlines() if line.strip()]
        tokens = " \n ".join(lines).split(" ")
        if sum(t != "\n" for t in tokens) < 5:
            continue
        for start, end in split_words(tokens):
            text = " ".join(tokens[start:end]).replace(" \n ", "\n").strip("\n ")
            yield {"title": title, "location": heading, "text": text}


def chunk_pdf(path: Path):
    """Split a PDF into overlapping word windows. Location = page (or page range)."""
    title = title_from_filename(path)
    tagged = []  # (word, page_number)
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            for word in clean(page.get_text()).split():
                tagged.append((word, page_number))
    words = [w for w, _ in tagged]
    for start, end in split_words(words):
        first, last = tagged[start][1], tagged[end - 1][1]
        location = f"p. {first}" if first == last else f"pp. {first}-{last}"
        yield {"title": title, "location": location, "text": " ".join(words[start:end])}


def build_chunks(assistant_id: str):
    folder = KNOWLEDGE / assistant_id
    chunks = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in (".md", ".txt"):
            pieces = chunk_markdown(path)
        elif path.suffix.lower() == ".pdf":
            pieces = chunk_pdf(path)
        else:
            continue
        public = "public" in path.relative_to(folder).parts
        for i, piece in enumerate(pieces):
            piece.update({
                "id": f"{assistant_id}:{re.sub(r'[^A-Za-z0-9]+', '_', path.stem)}:{i}",
                "filename": path.name,
                "public": public,
            })
            chunks.append(piece)
    return chunks


def embedding_input(chunk) -> str:
    # The title and section travel with the text so a chunk like "Retainers:
    # $5,000..." still matches a question that names the document.
    return f"{chunk['title']} | {chunk['location']}\n{chunk['text']}"


def embed_all(client: OpenAI, texts, batch=100):
    vectors = []
    for i in range(0, len(texts), batch):
        response = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        vectors.extend(d.embedding for d in response.data)
        print(f"   embedded {min(i + batch, len(texts))}/{len(texts)}")
    return vectors


def main():
    assistants = json.loads((ROOT / "assistants.json").read_text(encoding="utf-8"))
    targets = sys.argv[1:] or list(assistants)
    unknown = [t for t in targets if t not in assistants]
    if unknown:
        sys.exit(f"Unknown assistant(s): {', '.join(unknown)}. Add them to assistants.json first.")

    all_chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8")) if CHUNKS_FILE.exists() else {}
    client = OpenAI()
    index = Pinecone().Index(PINECONE_INDEX)

    for assistant_id in targets:
        print(f"== {assistant_id}")
        chunks = build_chunks(assistant_id)
        if not chunks:
            sys.exit(f"No documents found in knowledge/{assistant_id}/")
        print(f"   {len(chunks)} chunks from {len({c['filename'] for c in chunks})} files")
        vectors = embed_all(client, [embedding_input(c) for c in chunks])

        try:
            index.delete(delete_all=True, namespace=assistant_id)
        except Exception as e:  # a namespace that does not exist yet raises on some versions
            print(f"   (namespace was empty: {e.__class__.__name__})")
        index.upsert(
            vectors=[{"id": c["id"], "values": v, "metadata": {"filename": c["filename"]}}
                     for c, v in zip(chunks, vectors)],
            namespace=assistant_id,
            batch_size=100,
            show_progress=False,
        )
        all_chunks[assistant_id] = chunks
        print(f"   uploaded {len(vectors)} vectors to namespace '{assistant_id}'")

    CHUNKS_FILE.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {CHUNKS_FILE.name}")


if __name__ == "__main__":
    main()
