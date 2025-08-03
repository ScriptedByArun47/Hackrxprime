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
    print("🔥 Warming up Gemini model and loading FAISS...")

    try:
        # --- Load clauses from clause cache ---
        clause_file_path = os.path.abspath(os.path.join( "clause_cache", "6635d94cf9023c83521982b3043ec70c.json"))
        if not os.path.exists(clause_file_path):
            print("⚠️ Clause file not found:", clause_file_path)
            return

        with open(clause_file_path, "r", encoding="utf-8") as f:
            raw_clauses = json.load(f)
            print(f"🗂 Loaded clauses from cache file: {clause_file_path}")

        # --- Filter valid clauses ---
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
            print("⚠️ No valid clauses found in file, skipping FAISS index build")
        else:
            # --- Build FAISS index ---
            clause_embeddings = model.encode(clause_texts, show_progress_bar=False)
            index = faiss.IndexFlatL2(clause_embeddings.shape[1])
            index.add(np.array(clause_embeddings))

            app.state.index = index
            app.state.clauses = valid_clauses
            print(f"✅ FAISS index loaded with {len(clause_texts)} clauses")

        # --- Gemini LLM warmup ---
        sample_question = "What is covered under hospitalization?"
        sample_clause = "Hospitalization covers room rent, nursing charges, and medical expenses incurred due to illness or accident."

        tokens = len(tokenizer.tokenize(sample_clause))
        trimmed_clause = [{"clause": sample_clause}] if tokens < 512 else []

        qmap = {sample_question: trimmed_clause}
        prompt = build_prompt_batch(qmap)

        result = await call_llm(prompt, 0, 1)
        print("✅ Gemini warmup complete:", result.get("Q1", {}).get("answer"))

    except Exception as e:
        print("❌ Warmup failed:", str(e))



class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

def build_faiss_index(clauses: List[Dict]) -> tuple:
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, texts

def extract_keywords(question: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', question.lower())
    stopwords = {"what", "is", "the", "of", "under", "a", "an", "how", "for", "and", "in", "on", "to", "does", "do", "are"}
    return [t for t in tokens if t not in stopwords and len(t) > 2]

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
    

def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def save_clause_cache(url: str, clauses: List[Dict[str, str]]):
    os.makedirs("clause_cache", exist_ok=True)
    cache_path = f"clause_cache/{url_hash(url)}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)
    print(f"✅ Clause cache saved to: {cache_path}")

@app.post("/api/v1/hackrx/run")
async def hackrx_run(req: HackRxRequest):
        # ✅ Use preloaded FAISS index if available from startup
    if hasattr(app.state, "index") and hasattr(app.state, "clauses"):
        index = app.state.index
        clause_texts = [c["clause"] for c in app.state.clauses]
    else:
        # Fallback to runtime clause extraction if FAISS not loaded
        doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]
        all_clauses = []
        from pathlib import Path

        for url in doc_urls:
            try:
                cache_path = f"clause_cache/{url_hash(url)}.json"
                if Path(cache_path).exists():
                    with open(cache_path, "r", encoding="utf-8") as f:
                        clauses = json.load(f)
                    print(f"🔁 Loaded cached clauses for {url}")
                else:
                    clauses = extract_clauses_from_url(url)
                    save_clause_cache(url, clauses)
                    print(f"📄 Extracted and cached clauses for {url}")
                all_clauses.extend(clauses)
            except Exception as e:
                print(f"❌ Failed to extract from URL {url}:", e)

        index, clause_texts = build_faiss_index(all_clauses)



    question_clause_map = {}
    uncached_questions = []
    for question in req.questions:
        if question in qa_cache:
            continue
        top = get_top_clauses(question, index, clause_texts)
        trimmed = trim_clauses([{"clause": c} for c in top])
        question_clause_map[question] = trimmed
        uncached_questions.append(question)

    # Batch uncached questions
    batch_size = 50
    batches = [list(question_clause_map.items())[i:i + batch_size] for i in range(0, len(uncached_questions), batch_size)]
    prompts = [build_prompt_batch(dict(batch)) for batch in batches]

    tasks = [call_llm(prompt, i * batch_size, len(batch)) for i, (prompt, batch) in enumerate(zip(prompts, batches))]
    results = await asyncio.gather(*tasks)

    merged = {}
    for result in results:
        merged.update(result)

    # Save new answers to cache
    for i, question in enumerate(uncached_questions):
        answer = merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        qa_cache[question] = answer

    # Persist updated cache
    with open(QA_CACHE_FILE, "w") as f:
        json.dump(qa_cache, f, indent=2)

    # Build final answer list
    final_answers = [
        qa_cache.get(q) if q in qa_cache else merged.get(f"Q{i+1}", {}).get("answer", "No answer found.")
        for i, q in enumerate(req.questions)
    ]
    return {"answers": final_answers}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
