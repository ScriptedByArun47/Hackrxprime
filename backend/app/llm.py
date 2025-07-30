# app/llm.py

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE, build_mistral_prompt, build_batch_prompt

# 🔐 Load and configure Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API")
genai.configure(api_key=api_key)

# Use Gemini 1.5 Flash for low-latency answers
genai_model = genai.GenerativeModel("models/gemini-1.5-flash")

# 🧼 Sanitize raw Gemini output
def _sanitize_llm_output(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    return raw.strip("`").strip()

# 🔹 Answer one question using clause context
def query_mistral_with_clauses(question: str, clauses: list) -> dict:
    prompt = build_mistral_prompt(question, clauses)

    try:
        response = genai_model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "top_p": 0.7,
                "max_output_tokens": 100,
            }
        )
        clean = _sanitize_llm_output(response.text)
        return json.loads(clean)

    except json.JSONDecodeError:
        return {
            "answer": "The document does not contain a clear or relevant clause to address this query. Please consult the insurer or full policy."
        }

    except Exception as e:
        print(f"❌ Error in query_mistral_with_clauses: {e}")
        return {
            "answer": "LLM processing error. Please try again.",
            "supporting_clause": "None",
            "explanation": str(e)
        }

# 🔹 Answer multiple questions at once
def query_mistral_batch(questions: list, clauses: list) -> dict:
    prompt = build_batch_prompt(questions, clauses)

    try:
        response = genai_model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "top_p": 0.7,
                "max_output_tokens": 150,
            }
        )
        clean = _sanitize_llm_output(response.text)
        return json.loads(clean)

    except json.JSONDecodeError:
        return {f"Q{i+1}": "Invalid or incomplete answer." for i in range(len(questions))}

    except Exception as e:
        print(f"❌ Error in query_mistral_batch: {e}")
        return {f"Q{i+1}": "LLM processing error." for i in range(len(questions))}
