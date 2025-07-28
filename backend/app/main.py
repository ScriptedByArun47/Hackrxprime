# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from app.extract_clauses import extract_clauses_from_url
from app.prompts import build_mistral_prompt
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import asyncio
import re
import google.generativeai as genai
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os


# 🔧 Setup
load_dotenv() 
api_key = os.getenv("GEMINI_API")
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
genai.configure(api_key=api_key)
genai_model = genai.GenerativeModel('models/gemini-1.5-flash')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

def build_faiss_index(clauses):
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, texts

def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {"what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on", "to", "does", "do", "are"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]

def get_top_clauses(question, index, texts, k=15):
    q_vector = model.encode([question])
    _, I = index.search(np.array(q_vector), k)
    top_clauses = [texts[i] for i in I[0]]
    keywords = extract_keywords(question)
    keyword_matches = [c for c in texts if any(k in c.lower() for k in keywords)]
    combined = list(dict.fromkeys(top_clauses + keyword_matches))

    def keyword_score(clause):
        return sum(1 for word in keywords if word in clause.lower())

    combined = sorted(combined, key=keyword_score, reverse=True)
    return combined[:5]  # 🔧 reduced from 10 → 5

def trim_clauses(clauses, max_tokens=1200):
    result = []
    total = 0
    for c in clauses:
        clause = c["clause"]
        tokens = len(tokenizer.tokenize(clause))
        if total + tokens > max_tokens:
            break
        result.append({"clause": clause})
        total += tokens
    return result

def build_prompt_batch(question_clause_map: dict) -> str:
    question_json_lines = []
    for i, (q, clauses) in enumerate(question_clause_map.items(), 1):
        joined = " ".join(c["clause"] for c in clauses)
        question_json_lines.append(f'"Q{i}": {{ "question": "{q}", "clauses": """{joined}""" }}')
    batched_input = "{\n" + ",\n".join(question_json_lines) + "\n}"
    prompt = (
        "You are an expert insurance assistant. For each entry below, answer the question based strictly on the clauses.\n"
        "Respond as JSON with keys 'Q1', 'Q2', ..., each mapping to an object with an 'answer' key like this:\n"
        '{ "Q1": {"answer": "..."}, "Q2": {"answer": "..."} }\n\n'
        "Entries:\n" + batched_input
    )
    return prompt

async def call_llm(prompt: str, offset: int, batch_size: int):
    try:
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={"response_mime_type": "application/json"},
        )
        content = response.text.strip()
        if content.startswith("json"):
            content = content.replace("json", "").replace("```", "").strip()
        parsed = json.loads(content)
        # ✅ Print token count
        if hasattr(response, "usage_metadata"):
            print(f"🔢 Total tokens used in batch {offset//batch_size + 1}: {response.usage_metadata.total_token_count}")
            
        return {f"Q{offset + i + 1}": parsed.get(f"Q{i+1}", {"answer": "No answer"}) for i in range(batch_size)}
    except Exception as e:
        print("❌ LLM Error:", e)
        return {f"Q{offset + i + 1}": {"answer": "Error"} for i in range(batch_size)}

@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]
    all_clauses = []
    for url in doc_urls:
        all_clauses.extend(extract_clauses_from_url(url))

    index, clause_texts = build_faiss_index(all_clauses)

    question_clause_map = {}
    for q in req.questions:
        top = get_top_clauses(q, index, clause_texts)
        trimmed = trim_clauses([{"clause": c} for c in top])
        question_clause_map[q] = trimmed

    # Split into 5 batches
    batch_size = (len(req.questions) + 4) // 5
    batches = [
        list(question_clause_map.items())[i:i + batch_size]
        for i in range(0, len(req.questions), batch_size)
    ]

    prompts = [build_prompt_batch(dict(batch)) for batch in batches]

    tasks = [call_llm(prompt, i * batch_size, len(batch)) for i, (prompt, batch) in enumerate(zip(prompts, batches))]
    results = await asyncio.gather(*tasks)

    merged = {}
    for r in results:
        merged.update(r)

    answers = [merged.get(f"Q{i+1}", {}).get("answer", "Error") for i in range(len(req.questions))]
    return {"answers": answers}
