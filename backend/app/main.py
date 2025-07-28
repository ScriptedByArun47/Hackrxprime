# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from app.extract_clauses import extract_clauses_from_url
from app.prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import requests
import asyncio
import sys
import os
import re  # 🔧 added for keyword extraction
from app.prompts import build_mistral_prompt

import google.generativeai as genai

app = FastAPI()

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

model = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key="AIzaSyD1nYVMEV5c5KWrO9cMpzPUYPSByM3wt00")

genai_model = genai.GenerativeModel('models/gemini-1.5-flash')


def build_faiss_index(clauses):
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, texts, vectors

# 🔧 keyword extractor
def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {"what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on", "to", "does", "do", "are"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]

# 🔧 FAISS + keyword relevance hybrid retrieval
def get_top_clauses(question, index, texts, k=15):
    q_vector = model.encode([question])
    _, I = index.search(np.array(q_vector), k)
    top_clauses = [texts[i] for i in I[0]]

    # Add keyword matching clauses
    keywords = extract_keywords(question)
    keyword_matches = [c for c in texts if any(k in c.lower() for k in keywords)]

    # Merge + deduplicate
    combined = list(dict.fromkeys(top_clauses + keyword_matches))  # preserves order

    # Re-rank by simple keyword overlap
    def keyword_score(clause):
        return sum(1 for word in keywords if word in clause.lower())

    combined = sorted(combined, key=keyword_score, reverse=True)
    return combined[:10]


async def call_genai_llm_async(prompt: str, timeout: int = 120) -> dict:
    try:
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={
                "response_mime_type": "application/json"
            }
        )

        raw_output = response.text.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        usage = getattr(response, "usage_metadata", None)
        if usage:
            print(f"🔹 Prompt tokens: {usage.prompt_token_count}")
            print(f"🔹 Response tokens: {usage.candidates_token_count}")

        return json.loads(raw_output)

    except Exception as e:
        print(f"❌ Error calling GenAI API: {e}")
        return {
            "answer": "Error",
            "supporting_clause": "None",
            "explanation": f"Error while calling LLM API: {str(e)}"
        }

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def trim_clauses(clauses, max_tokens=1800):
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


@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]

    all_clauses = []
    for url in doc_urls:
        all_clauses.extend(extract_clauses_from_url(url))

    index, clause_texts, _ = build_faiss_index(all_clauses)

    async def process_question(q):
        print(f"\n🧠 Processing question: {q}")
        top_clauses_raw = get_top_clauses(q, index, clause_texts, k=15)  # 🔧 raised k to 15 for more breadth
        print(f"📌 Top clauses: {top_clauses_raw[:2]}")
        clause_objects = trim_clauses([{"clause": c} for c in top_clauses_raw])
        print(f"✂️ Trimmed {len(clause_objects)} clauses")

        prompt = build_mistral_prompt(q, clause_objects, max_tokens=1800)
        print(f"📝 Final prompt:\n{prompt[:300]}...")
        response = await call_genai_llm_async(prompt)
        print(f"📥 LLM response: {response}")
        return response

    results = await asyncio.gather(*[process_question(q) for q in req.questions])
    return {"answers": [res.get("answer", "No answer found") for res in results]}
