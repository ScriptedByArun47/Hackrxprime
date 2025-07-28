import os

if os.path.exists("app/data/faiss.index"):
    self.index = faiss.read_index("app/data/faiss.index")
else:
    self.index, self.embeddings = self.build_index()
