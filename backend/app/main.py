# [Unchanged top imports]
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
import hashlib
import time

# Load env vars and API key
load_dotenv()
api_key = os.getenv("GEMINI_API")
genai.configure(api_key=api_key)

# FastAPI app
app = FastAPI()

# Load embedding model and tokenizer
model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
genai_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# QA cache preload
QA_CACHE_FILE = "qa_cache.json"
qa_cache = {}
if os.path.exists(QA_CACHE_FILE):
    with open(QA_CACHE_FILE, "r", encoding="utf-8") as f:
        try:
            qa_cache = json.load(f)
            print(f"✅ Loaded QA cache with {len(qa_cache)} entries")
        except json.JSONDecodeError:
            print("⚠️ QA cache is corrupted. Starting fresh.")
            qa_cache = {}

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def save_clause_cache(url: str, clauses: List[Dict[str, str]]):
    os.makedirs("clause_cache", exist_ok=True)
    cache_path = f"clause_cache/{url_hash(url)}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)

def is_probably_insurance_policy(clauses: List[Dict[str, str]]) -> bool:
    insurance_keywords = {
        "coverage", "hospitalization", "sum insured", "premium", "pre-existing",
        "benefit", "exclusion", "waiting period", "treatment", "illness", "policy"
    }
    match_count = 0
    for clause in clauses[:40]:
        text = clause.get("clause", "").lower()
        matches = sum(1 for word in insurance_keywords if word in text)
        if matches >= 2:
            match_count += 1
    return match_count >= 5

def build_faiss_index(clauses: List[Dict]) -> tuple:
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors).astype(np.float32))
    return index, texts

def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {"what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on", "to", "does", "do", "are"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]

def trim_clauses(clauses: List[Dict[str, str]], max_tokens: int = 1000) -> List[Dict[str, str]]:
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

def build_prompt_batch(question_clause_map: Dict[str, List[Dict[str, str]]]) -> str:
    prompt_lines = []
    for i, (question, clauses) in enumerate(question_clause_map.items(), start=1):
        joined = " ".join(c["clause"].replace('\\', '\\\\').replace('"', '\\"') for c in clauses)
        prompt_lines.append(f'"Q{i}": {{"question": "{question}", "clauses": "{joined}"}}')
    json_data = "{\n" + ",\n".join(prompt_lines) + "\n}"
    return (
        "You are an expert insurance assistant. Answer the following questions based strictly on the provided clauses.\n"
        "Respond only in JSON with keys like 'Q1', 'Q2', each containing an 'answer'.\n"
        "Example format:\n"
        '{ "Q1": {"answer": "..."}, "Q2": {"answer": "..."} }\n\n'
        f"Entries:\n{json_data}"
    )

async def call_llm(prompt: str, offset: int, batch_size: int) -> Dict[str, Dict[str, str]]:
    try:
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={"response_mime_type": "application/json"},
        )
        content = getattr(response, "text", None) or response.candidates[0].content.parts[0].text
        content = content.strip().lstrip("```json").rstrip("```").strip()
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

@app.on_event("startup")
async def warmup_model():
    print("🔥 Warming up Gemini model and loading FAISS...")
    app.state.cache_indices = {}

    clause_dir = "clause_cache"
    for filename in os.listdir(clause_dir):
        if filename.endswith(".json"):
            path = os.path.join(clause_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_clauses = json.load(f)
            except Exception:
                print(f"❌ Failed to load {filename}")
                continue

            valid_clauses = []
            clause_texts = []

            for item in raw_clauses:
                clause = item.get("clause", "").strip()
                if clause:
                    tokens = len(tokenizer.tokenize(clause))
                    if tokens <= 512:
                        valid_clauses.append({"clause": clause})
                        clause_texts.append(clause)

            if not clause_texts:
                print(f"⚠️ No valid clauses in {filename}")
                continue

            embeddings = model.encode(clause_texts, show_progress_bar=False)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype(np.float32))
            urlhash = filename.replace(".json", "")
            app.state.cache_indices[urlhash] = {
                "index": index,
                "clauses": valid_clauses
            }
            print(f"✅ Loaded FAISS index for {filename} with {len(clause_texts)} clauses")

    try:
        sample_question = "What is covered under hospitalization?"
        sample_clause = "Hospitalization covers room rent, nursing charges, and medical expenses incurred due to illness or accident."
        if len(tokenizer.tokenize(sample_clause)) < 512:
            prompt = build_prompt_batch({sample_question: [{"clause": sample_clause}]})
            result = await call_llm(prompt, 0, 1)
            print("✅ Gemini warmup complete:", result.get("Q1", {}).get("answer"))
    except Exception as e:
        print("❌ Gemini warmup failed:", e)

@app.post("/api/v1/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    from pathlib import Path
    start_time = time.time()

    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]
    all_clauses = []

    for url in doc_urls:
        try:
            cache_path = f"clause_cache/{url_hash(url)}.json"
            if Path(cache_path).exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    clauses = json.load(f)
                print(f"🔁 Loaded cached clauses for {url}")
            else:
                clauses = extract_clauses_from_url(url)
                if clauses and is_probably_insurance_policy(clauses):
                    save_clause_cache(url, clauses)
                    print(f"📄 Extracted and cached clauses for {url}")
                    all_clauses.extend(clauses)
                else:
                    print(f"⚠️ Skipping non-insurance document: {url}")
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                        print(f"🧹 Removed stale cache file: {cache_path}")
                    continue
            all_clauses.extend(clauses)
        except Exception as e:
            print(f"❌ Failed to extract from URL {url}:", e)

    if not all_clauses or not doc_urls:
        return {"answers": ["No valid clauses found in provided documents."] * len(req.questions)}

    url0_hash = url_hash(doc_urls[0])
    if url0_hash in app.state.cache_indices:
        print(f"⚡ Using preloaded FAISS index for {url0_hash}")
        index = app.state.cache_indices[url0_hash]["index"]
        clause_texts = [c["clause"] for c in app.state.cache_indices[url0_hash]["clauses"]]
    else:
        valid_clauses = [c for c in all_clauses if c.get("clause", "").strip()]
        clause_texts = [c["clause"] for c in valid_clauses]
        index, _ = build_faiss_index(valid_clauses)
        app.state.cache_indices[url0_hash] = {
            "index": index,
            "clauses": valid_clauses
        }

    t1 = time.time()
    uncached_questions = [q for q in req.questions if q not in qa_cache]
    question_clause_map = {}

    if uncached_questions:
        question_embeddings = model.encode(uncached_questions, batch_size=32, show_progress_bar=False)
        _, indices = index.search(np.array(question_embeddings).astype(np.float32), k=15)

        for i, question in enumerate(uncached_questions):
            top_clauses = [clause_texts[j] for j in indices[i]]
            keywords = extract_keywords(question)
            keyword_matches = [c for c in clause_texts if any(k in c.lower() for k in keywords)]
            combined = list(dict.fromkeys(top_clauses + keyword_matches))
            sorted_clauses = sorted(combined, key=lambda clause: sum(1 for word in keywords if word in clause.lower()), reverse=True)[:7]
            # Dynamically adjust max tokens per question based on batch size
            per_question_token_limit = max(30000 // len(req.questions) - 500, 400)
            trimmed = trim_clauses([{"clause": c} for c in sorted_clauses], max_tokens=per_question_token_limit)

            question_clause_map[question] = trimmed
    print(f"🕒 Clause selection took {time.time() - t1:.2f} seconds")

    t2 = time.time()
    batch_size = 30
    batches = [list(question_clause_map.items())[i:i + batch_size] for i in range(0, len(uncached_questions), batch_size)]
    prompts = [build_prompt_batch(dict(batch)) for batch in batches]
    tasks = [call_llm(prompt, i * batch_size, len(batch)) for i, (prompt, batch) in enumerate(zip(prompts, batches))]
    results = await asyncio.gather(*tasks)
    print(f"🕒 Gemini response took {time.time() - t2:.2f} seconds")

    merged = {}
    for result in results:
        merged.update(result)
    for i, question in enumerate(uncached_questions):
        answer = merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        qa_cache[question] = answer

    t3 = time.time()
    with open(QA_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_cache, f, indent=2)
    print(f"🕒 Writing cache took {time.time() - t3:.2f} seconds")

    final_answers = [
        qa_cache.get(q) if q in qa_cache else merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        for i, q in enumerate(req.questions)
    ]
    print(f"✅ Total /run latency: {time.time() - start_time:.2f} seconds")
    return {"answers": final_answers}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
