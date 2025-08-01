import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CLAUSE_FILE = "app/data/clauses.json"
INDEX_FILE = "app/data/faiss.index"

class ClauseRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.clauses = self._load_clauses()
        self.index, _ = self._load_or_build_index()

    def _load_clauses(self):
        try:
            with open(CLAUSE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [c for c in data if "clause" in c and c["clause"].strip()]
        except Exception as e:
            print(f"❌ Error loading clauses: {e}")
            return []

    def _load_or_build_index(self):
        texts = [c["clause"] for c in self.clauses]
        if not texts:
            print("⚠️ No clauses found to build index.")
            return None, None

        if os.path.exists(INDEX_FILE):
            try:
                index = faiss.read_index(INDEX_FILE)
                return index, None
            except Exception as e:
                print(f"⚠️ Failed to load FAISS index from disk: {e}")

        embeddings = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)

        return index, embeddings

    def search(self, query: str, top_k: int = 5):
        if not self.index or not self.clauses:
            print("⚠️ Search failed: index or clauses not available.")
            return []

        query = query.strip().lower()
        query_embedding = np.array(
            self.model.encode([query], convert_to_numpy=True), dtype=np.float32
        )
        D, I = self.index.search(query_embedding, top_k)

        return [self.clauses[i] for i in I[0] if 0 <= i < len(self.clauses)]

    def warmup(self):
        _ = self.model.encode(["warmup query"], convert_to_numpy=True)

