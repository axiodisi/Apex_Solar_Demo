# smart_upload_pdfs.py - Complete working version
import os
import json
import hashlib
import re
from datetime import datetime
from pinecone import Pinecone
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

def clean_filename_for_id(filename):
    """Clean filename to be ASCII-only for Pinecone vector IDs"""
    # Remove all non-ASCII characters (keep only characters 0-127)
    cleaned = ''.join(char for char in filename if ord(char) < 128)
    
    # Replace any non-alphanumeric character with underscore
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', cleaned)
    
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned

def get_file_hash(file_path):
    """Get MD5 hash of file to detect changes"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_processed_files():
    """Load record of already processed files"""
    try:
        with open('processed_files.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_processed_files(processed_files):
    """Save record of processed files"""
    with open('processed_files.json', 'w') as f:
        json.dump(processed_files, indent=2, fp=f)

def smart_upload_pdfs():
    # Connect to services
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("apex-solar-demo")    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load record of processed files
    processed_files = load_processed_files()
    
    # Get all PDFs
    pdf_files = list(Path(r"C:\AI_Projects\Apex_Solar_Demo\knowledge_source").glob("*.txt"))    
    print(f"Found {len(pdf_files)} PDFs")
    
    new_vectors = []
    
    for pdf_file in pdf_files:
        file_path = str(pdf_file)
        current_hash = get_file_hash(file_path)
        
        # Check if file is new or changed
        if file_path in processed_files and processed_files[file_path]['hash'] == current_hash:
            print(f"⏩ Skipping unchanged: {pdf_file.name}")
            continue
        
        print(f"📄 Processing new/changed: {pdf_file.name}")
        
        # If file was previously processed, delete old vectors first
        if file_path in processed_files:
            old_chunk_ids = processed_files[file_path]['chunk_ids']
            print(f"   🗑️  Removing {len(old_chunk_ids)} old vectors...")
            index.delete(ids=old_chunk_ids)
        
        # Extract text
        # Extract text from .txt file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
            # Chunk text
            chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
        
        # Create vectors for this file
        chunk_ids = []
        file_vectors = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{clean_filename_for_id(pdf_file.stem)}_{current_hash[:8]}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            response = client.embeddings.create(input=chunk, model="text-embedding-3-small")
            embedding = response.data[0].embedding
            
            file_vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {"text": chunk, "filename": pdf_file.name}
            })
        
        new_vectors.extend(file_vectors)
        
        # Update processed files record
        processed_files[file_path] = {
            "hash": current_hash,
            "chunk_ids": chunk_ids,
            "processed_date": str(datetime.now()),
            "chunk_count": len(chunks)
        }
        
        print(f"   ✅ Prepared {len(chunks)} chunks")
    
    # Remove deleted files from tracking
    current_file_paths = {str(f) for f in pdf_files}
    for tracked_file in list(processed_files.keys()):
        if tracked_file not in current_file_paths:
            print(f"🗑️  File deleted, removing vectors: {Path(tracked_file).name}")
            index.delete(ids=processed_files[tracked_file]['chunk_ids'])
            del processed_files[tracked_file]
    
    # Upload new vectors in batches
    if new_vectors:
        print(f"\n🔄 Uploading {len(new_vectors)} new vectors...")
        batch_size = 100
        
        for i in range(0, len(new_vectors), batch_size):
            batch = new_vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"   ✅ Uploaded batch {i//batch_size + 1} ({len(batch)} vectors)")
    else:
        print("✅ No new files to process!")
    
    # Save updated tracking file
    save_processed_files(processed_files)
    
    print(f"\n🎉 Smart upload complete!")
    print(f"   📊 Total files tracked: {len(processed_files)}")
    print(f"   📈 New vectors uploaded: {len(new_vectors)}")

if __name__ == "__main__":
    smart_upload_pdfs()