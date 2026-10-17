# settings.py - shared configuration, overridable with environment variables on Render.
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")  # local development only; Render sets real environment variables

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "apex-solar-demo")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
DEFAULT_ASSISTANT = os.getenv("DEFAULT_ASSISTANT", "law-firm")

# Sites allowed to call the API from a browser.
ALLOWED_ORIGINS = [
    "https://roccocolasacco.com",
    "https://www.roccocolasacco.com",
    "http://localhost:3000",
]
ALLOWED_ORIGIN_REGEX = r"https://.*\.vercel\.app|http://localhost:\d+"

# Per-visitor limit to protect the OpenAI bill.
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "600"))
