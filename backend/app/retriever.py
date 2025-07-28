# app/retriever.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CLAUSE_FILE = "app/data/clauses.json"

class ClauseRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.clauses = self.load_clauses()
        self.index, self.embeddings = self.build_index()

    def load_clauses(self):
        try:
            with open(CLAUSE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading clauses: {e}")
        return []


    def build_index(self):
        texts = [c["clause"] for c in self.clauses]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    # ✅ Save index to disk
        faiss.write_index(index, "app/data/faiss.index")

        return index, embeddings


    def search(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(query_embedding, top_k)
        return [self.clauses[idx] for idx in I[0]]