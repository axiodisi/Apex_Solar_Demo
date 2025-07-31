# main.py - FastAPI with Pinecone (converted from ChromaDB)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import os
from pinecone import Pinecone
from openai import OpenAI
import uvicorn
import uuid
import time
#from dotenv import load_dotenv

# Load environment variables
#load_dotenv()

# === Setup ===
app = FastAPI(title="Apex Solar Demo API")

# Add CORS for your React website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global Pinecone index (will be initialized on startup)
pinecone_index = None

@app.on_event("startup")
async def startup_event():
    global pinecone_index
    
    print("🚀 Starting RAG API...")
    print("🔑 Checking environment variables...")
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_key:
        print("❌ PINECONE_API_KEY not found!")
        return
    if not openai_key:
        print("❌ OPENAI_API_KEY not found!")
        return
        
    print("✅ Environment variables found")
    
    try:
        print("🔄 Connecting to Pinecone...")
        pc = Pinecone(api_key=pinecone_key)
        
        # --- CHANGED: Updated the index name ---
        index_name = "apex-solar-demo"
        print(f"🔄 Getting index: {index_name}")
        pinecone_index = pc.Index(index_name)
        print("✅ Connected to Pinecone successfully")
        
        print("🔄 Testing index stats...")
        stats = pinecone_index.describe_index_stats()
        print(f"📊 Index has {stats.total_vector_count} vectors")
        
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        print("❌ API will run without Pinecone connection")

# === Utils ===
def extract_text_from_pdf(file: UploadFile):
    """Extract text from uploaded PDF file"""
    doc = fitz.open(stream=file.file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI for multiple texts"""
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text, 
            model="text-embedding-3-small"
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            space_pos = text.rfind(' ', start, end)
            if space_pos > start + chunk_size // 2:
                end = space_pos
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks

# === Models ===
class Query(BaseModel):
    question: str

class UploadResponse(BaseModel):
    status: str
    chunks: int
    filename: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# === Endpoints ===

@app.get("/")
def read_root():
    return {
        "status": "RAG API is running",
        "message": "This is the backend for the Apex Solar demo.",
        "pinecone_connected": pinecone_index is not None
    }

@app.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document file"""
    global pinecone_index
    
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone not connected")
    
    try:
        if file.filename.endswith(".pdf"):
            content = extract_text_from_pdf(file)
        else:
            file_content = await file.read()
            content = file_content.decode('utf-8')
        
        chunks = chunk_text(content)
        embeddings = embed_texts(chunks)
        
        vectors_to_upload = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{file.filename}_{i}_{str(uuid.uuid4())[:8]}"
            vectors_to_upload.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"text": chunk, "filename": file.filename}
            })
        
        pinecone_index.upsert(vectors=vectors_to_upload)
        
        return UploadResponse(status="success", chunks=len(chunks), filename=file.filename)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def ask_question(q: Query):
    """Query the knowledge base"""
    global pinecone_index
    
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone not connected")
    
    try:
        question_embedding = embed_texts([q.question])[0]
        
        search_results = pinecone_index.query(
            vector=question_embedding,
            top_k=5, # Reduced top_k for more focused context
            include_metadata=True
        )
        
        context_chunks = []
        sources = []
        for match in search_results.matches:
            if match.score > 0.3: # Increased threshold for higher relevance
                context_chunks.append(match.metadata['text'])
                sources.append(match.metadata.get('filename', 'Unknown'))
                
        if not context_chunks:
            return QueryResponse(answer="I couldn't find a relevant answer in the Apex Solar knowledge base. Please try rephrasing your question.", sources=[])
        
        context = "\n\n---\n\n".join(context_chunks)
        
        # --- IMPROVED: More specific and professional prompt ---
        prompt = f"""You are an internal assistant for a company called 'Apex Solar Solutions'. Answer the user's question based strictly on the provided context. Do not make information up. If the answer is not in the context, state that you cannot find the answer in the provided documents. Be concise and direct.

Context from internal documents:
{context}

Question: {q.question}

Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.0 # Set to 0 for factual, deterministic answers
        )
        
        return QueryResponse(
            answer=response.choices[0].message.content,
            sources=list(set(sources))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))