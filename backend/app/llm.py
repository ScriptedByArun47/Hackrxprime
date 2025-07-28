import json
import os
import google.generativeai as genai
from prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE, build_mistral_prompt, build_batch_prompt

genai.configure(api_key="AIzaSyAxtoi-r_gHKppRIo9tNhoOUS9akbf8qhg")
genai_model = genai.GenerativeModel('models/gemini-1.5-flash')

# Escape dangerous characters to avoid JSON issues
def _sanitize_llm_output(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    return raw.strip("`").strip()

# 🔹 Single Question LLM Call
def query_mistral_with_clauses(query, clauses):
    prompt = build_mistral_prompt(query, clauses)

    try:
        response = genai_model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "top_p": 0.7,
                "max_output_tokens": 100
            }
        )
        content = _sanitize_llm_output(response.text)
        return json.loads(content)

    except json.JSONDecodeError:
        return {
            "answer": "The document does not contain any clear or relevant clause to address the query. Please refer to the policy document directly or contact the insurer."
        }
    except Exception as e:
        print(f"❌ Error calling GenAI API from llm.py: {e}")
        return {
            "answer": "Error",
            "supporting_clause": "None",
            "explanation": f"Error while calling LLM API: {str(e)}"
        }

# 🔹 Batch Gemini Query (used in /hackrx/run)
def query_mistral_batch(questions, clauses):
    prompt = build_batch_prompt(questions, clauses)

    try:
        response = genai_model.generate_content(
            contents=[{"role": "user", "parts": [prompt]}],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "top_p": 0.7,
                "max_output_tokens": 150
            }
        )
        content = _sanitize_llm_output(response.text)
        return json.loads(content)

    except json.JSONDecodeError:
        return {f"Q{i+1}": "Invalid or incomplete answer." for i in range(len(questions))}
    except Exception as e:
        print(f"❌ Error in batch Gemini call: {e}")
        return {f"Q{i+1}": "Error" for i in range(len(questions))}
