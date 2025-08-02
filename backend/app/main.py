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
import time
from concurrent.futures import ThreadPoolExecutor

# Load env vars and API key
load_dotenv()
api_key = os.getenv("GEMINI_API")
genai.configure(api_key=api_key)

# FastAPI app
app = FastAPI()

class RunPayload(BaseModel):
    questions: List[str]

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

# QA cache file
QA_CACHE_FILE = "qa_cache.json"
qa_cache = {}
if os.path.exists(QA_CACHE_FILE):
    with open(QA_CACHE_FILE, "r") as f:
        try:
            qa_cache = json.load(f)
        except json.JSONDecodeError:
            qa_cache = {}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.on_event("startup")
async def warmup_model():
    print("🔥 Warming up Gemini model...")

    sample_question = "What is covered under hospitalization?"
    sample_clause = "Hospitalization covers room rent, nursing charges, and medical expenses incurred due to illness or accident."

    try:
        clause_vector = model.encode([sample_clause])
        index = faiss.IndexFlatL2(clause_vector.shape[1])
        index.add(np.array(clause_vector))

        tokens = len(tokenizer.tokenize(sample_clause))
        trimmed_clause = [{"clause": sample_clause}] if tokens < 512 else []

        qmap = {sample_question: trimmed_clause}
        prompt = build_prompt_batch(qmap)

        result = await call_llm(prompt, 0, 1)
        print("✅ Warmup complete:", result.get("Q1", {}).get("answer"))
    except Exception as e:
        print("❌ Warmup failed:", str(e))

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

def build_faiss_index(clauses: List[Dict]) -> tuple:
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, texts

def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {"what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on", "to", "does", "do", "are"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]

def get_top_clauses(question: str, index, texts: List[str], k: int = 15) -> List[str]:
    q_vector = model.encode([question], show_progress_bar=False)
    _, I = index.search(np.array(q_vector), k)
    top_clauses = [texts[i] for i in I[0]]
    keywords = extract_keywords(question)
    keyword_matches = [c for c in texts if any(k in c.lower() for k in keywords)]
    combined = list(dict.fromkeys(top_clauses + keyword_matches))

    def keyword_score(clause: str) -> int:
        return sum(1 for word in keywords if word in clause.lower())

    return sorted(combined, key=keyword_score, reverse=True)[:7]

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

def build_prompt_batch(question_clause_map: Dict[str, List[Dict[str, str]]]) -> str:
    prompt_lines = []
    for i, (question, clauses) in enumerate(question_clause_map.items(), start=1):
        joined = " ".join(c["clause"].replace('\\', '\\\\').replace('"', '\\"') for c in clauses)
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

@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    start = time.time()
    print("⏱️ Started /hackrx/run")

    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]
    all_clauses = []

    with ThreadPoolExecutor() as executor:
        tasks = [asyncio.to_thread(extract_clauses_from_url, url) for url in doc_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, res in zip(doc_urls, results):
            if isinstance(res, Exception):
                print(f"❌ Failed to extract from URL {url}:", res)
            else:
                all_clauses.extend(res)

    print(f"📄 Clause extraction completed in {time.time() - start:.2f}s")

    index, clause_texts = build_faiss_index(all_clauses)
    print(f"🔍 FAISS index built in {time.time() - start:.2f}s")

    question_clause_map = {}
    uncached_questions = []
    for question in req.questions:
        if question in qa_cache:
            continue
        top = get_top_clauses(question, index, clause_texts)
        trimmed = trim_clauses([{"clause": c} for c in top])
        question_clause_map[question] = trimmed
        uncached_questions.append(question)

    print(f"📌 Clause retrieval + trimming done in {time.time() - start:.2f}s")

    batch_size = max(1, (len(uncached_questions) + 4) // 5)
    batches = [list(question_clause_map.items())[i:i + batch_size] for i in range(0, len(uncached_questions), batch_size)]
    prompts = [build_prompt_batch(dict(batch)) for batch in batches]

    print(f"✍️ Prompt generation done in {time.time() - start:.2f}s")

    tasks = [call_llm(prompt, i * batch_size, len(batch)) for i, (prompt, batch) in enumerate(zip(prompts, batches))]
    print("⚡ Sending prompts to Gemini...")
    results = await asyncio.gather(*tasks)
    print(f"✅ Gemini responses received in {time.time() - start:.2f}s")

    merged = {}
    for result in results:
        merged.update(result)

    for i, question in enumerate(uncached_questions):
        answer = merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        qa_cache[question] = answer

    with open(QA_CACHE_FILE, "w") as f:
        json.dump(qa_cache, f, indent=2)

    final_answers = [
        qa_cache.get(q) if q in qa_cache else merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        for i, q in enumerate(req.questions)
    ]

    print(f"🏋️ Finished /hackrx/run in {time.time() - start:.2f}s")
    return {"answers": final_answers}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
