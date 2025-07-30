from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Dict
from fastapi.middleware.cors import CORSMiddleware
from app.extract_clauses import extract_clauses_from_url
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
import asyncio

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API")
genai.configure(api_key=api_key)

# FastAPI app
app = FastAPI()

# Embedding and tokenizer
model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
genai_model = genai.GenerativeModel('models/gemini-1.5-flash')

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

# Build FAISS index from clauses
def build_faiss_index(clauses: List[Dict]) -> tuple:
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, texts

# Keyword extractor for additional matching
def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {"what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on", "to", "does", "do", "are"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]

# Top clause retriever
def get_top_clauses(question: str, index, texts: List[str], k: int = 15) -> List[str]:
    q_vector = model.encode([question])
    _, I = index.search(np.array(q_vector), k)
    top_clauses = [texts[i] for i in I[0]]
    keywords = extract_keywords(question)
    keyword_matches = [c for c in texts if any(k in c.lower() for k in keywords)]
    combined = list(dict.fromkeys(top_clauses + keyword_matches))

    def keyword_score(clause: str) -> int:
        return sum(1 for word in keywords if word in clause.lower())

    return sorted(combined, key=keyword_score, reverse=True)[:7]

# Trim clauses based on token length
def trim_clauses(clauses: List[Dict[str, str]], max_tokens: int = 1200) -> List[Dict[str, str]]:
    result = []
    total = 0
    for clause_obj in clauses:
        clause = clause_obj["clause"]
        tokens = len(tokenizer.tokenize(clause))
        if total + tokens > max_tokens:
            break
        result.append({"clause": clause})
        total += tokens
    return result

# Prompt constructor
def build_prompt_batch(question_clause_map: Dict[str, List[Dict[str, str]]]) -> str:
    prompt_lines = []
    for i, (question, clauses) in enumerate(question_clause_map.items(), start=1):
        joined = " ".join(c["clause"].replace('"', '\\"') for c in clauses)
        prompt_lines.append(f'"Q{i}": {{"question": "{question}", "clauses": "{joined}"}}')

    json_data = "{\n" + ",\n".join(prompt_lines) + "\n}"
    prompt = (
        "You are an expert insurance assistant. Answer the following questions based strictly on the provided clauses.\n"
        "Respond only in JSON with keys like 'Q1', 'Q2', each containing an 'answer'.\n"
        "Example format:\n"
        '{ "Q1": {"answer": "..."}, "Q2": {"answer": "..."} }\n\n'
        f"Entries:\n{json_data}"
    )
    return prompt

# Gemini LLM call
async def call_llm(prompt: str, offset: int, batch_size: int) -> Dict[str, Dict[str, str]]:
    try:
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={"response_mime_type": "application/json"},
        )
        content = response.text.strip().lstrip("```json").rstrip("```").strip()
        parsed = json.loads(content)

        if hasattr(response, "usage_metadata"):
            print(f"🔢 Tokens used in batch {offset // batch_size + 1}: {response.usage_metadata.total_token_count}")

        return {
            f"Q{offset + i + 1}": parsed.get(f"Q{i + 1}", {"answer": "No answer found."})
            for i in range(batch_size)
        }
    except Exception as e:
        print("❌ LLM Error:", e)
        return {
            f"Q{offset + i + 1}": {"answer": "An error occurred while generating the answer."}
            for i in range(batch_size)
        }

# Main endpoint
@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]
    all_clauses = []

    for url in doc_urls:
        try:
            all_clauses.extend(extract_clauses_from_url(url))
        except Exception as e:
            print(f"❌ Failed to extract from URL {url}:", e)

    index, clause_texts = build_faiss_index(all_clauses)

    question_clause_map = {}
    for question in req.questions:
        top = get_top_clauses(question, index, clause_texts)
        trimmed = trim_clauses([{"clause": c} for c in top])
        question_clause_map[question] = trimmed

    # Batch into 5 groups
    batch_size = (len(req.questions) + 4) // 5
    batches = [list(question_clause_map.items())[i:i + batch_size] for i in range(0, len(req.questions), batch_size)]
    prompts = [build_prompt_batch(dict(batch)) for batch in batches]

    tasks = [call_llm(prompt, i * batch_size, len(batch)) for i, (prompt, batch) in enumerate(zip(prompts, batches))]
    results = await asyncio.gather(*tasks)

    merged = {}
    for result in results:
        merged.update(result)

    # Maintain original order
    final_answers = [merged.get(f"Q{i+1}", {}).get("answer", "No answer found.") for i in range(len(req.questions))]
    return {"answers": final_answers}

# Local dev runner
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
