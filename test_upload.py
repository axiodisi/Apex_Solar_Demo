import os
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# --- SETUP ---
# Load environment variables from your .env file
load_dotenv()

# Get the API keys from the environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "apex-solar-demo"

# Check if keys are loaded
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("❌ Error: Make sure PINECONE_API_KEY and OPENAI_API_KEY are in your .env file.")
    exit()

print("✅ API keys found.")

# --- THE TEST ---
try:
    # 1. Connect to Pinecone
    print("🔄 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print("✅ Connected to Pinecone index.")

    # 2. Connect to OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 3. Define a simple piece of test data
    test_text = "This is a test vector for Apex Solar."
    test_id = "test-vector-1"
    print(f"📝 Preparing to upload test vector with ID: {test_id}")

    # 4. Create an embedding for the test data
    print("🔄 Creating embedding with OpenAI...")
    response = client.embeddings.create(input=test_text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("✅ Embedding created.")

    # 5. Upload (Upsert) the single vector to Pinecone
    print("🔄 Uploading vector to Pinecone...")
    index.upsert(
        vectors=[
            {
                "id": test_id,
                "values": embedding,
                "metadata": {"text": test_text}
            }
        ]
    )
    print("✅ Vector uploaded successfully.")

    # 6. Verify the upload by checking the index stats
    print("🔄 Verifying upload by checking index stats...")
    stats = index.describe_index_stats()
    vector_count = stats.total_vector_count
    
    if vector_count > 0:
        print(f"\n🎉 SUCCESS! Vector count is now: {vector_count}")
    else:
        print(f"\n🤔 HMM... Upload reported success, but vector count is still 0. There might be a delay in indexing.")

except Exception as e:
    print(f"\n❌ FAILED: An error occurred.")
    print(f"   Error details: {e}")