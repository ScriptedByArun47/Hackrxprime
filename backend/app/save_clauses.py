# save_clauses.py
from app.extract_clauses import extract_clauses_from_url
from app.db import save_clauses_to_mongo
import sys

def save_clauses_from_url(url: str):
    clauses = extract_clauses_from_url(url)
    if not clauses:
        print(f"❌ No clauses found for URL: {url}")
        return
    clause_texts = [c["clause"] for c in clauses]
    save_clauses_to_mongo(clause_texts, url)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python save_clauses.py <document_url>")
    else:
        save_clauses_from_url(sys.argv[1])
