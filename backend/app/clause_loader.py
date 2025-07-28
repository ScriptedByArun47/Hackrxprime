import os
import numpy as np
import faiss
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from app.extract_clauses import extract_clauses_from_url

# Load embedding model
EMBEDDING_DIM = 384
model = SentenceTransformer("all-MiniLM-L6-v2")

# MongoDB setup
MONGO_URI = "mongodb+srv://kevin:Year2006@cluster0.c40cp0n.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # ✅ Hardcoded (no env)
client = MongoClient(MONGO_URI)
db = client["hackrx"]
collection = db["policy_clauses"]

# FAISS index
index = faiss.IndexFlatL2(EMBEDDING_DIM)

def embed_clauses(clauses):
    texts = [clause['clause'] for clause in clauses]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings

def preload(url):
    clauses = extract_clauses_from_url(url)
    embeddings = embed_clauses(clauses)

    index.add(np.array(embeddings).astype("float32"))

    for i, clause in enumerate(clauses):
        collection.insert_one({
            "faiss_id": index.ntotal - len(clauses) + i,
            "clause": clause['clause']
        })

    # ✅ Save FAISS index
    faiss.write_index(index, "app/data/faiss.index")

    print(f"✅ Loaded {len(clauses)} clauses into FAISS and MongoDB.")

# Example
if __name__ == "__main__":
    TEST_URL = "https://example.com/your-policy.pdf"
    preload(TEST_URL)
