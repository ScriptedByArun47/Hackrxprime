# llm.py
import json
import os
import google.generativeai as genai
from prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE, build_mistral_prompt

# Directly set your Google API key
genai.configure(api_key="AIzaSyD1nYVMEV5c5KWrO9cMpzPUYPSByM3wt00")

# Use the Gemini model (Gemma equivalent via Google)
genai_model = genai.GenerativeModel('models/gemini-1.5-flash')


def query_mistral_with_clauses(query, clauses):
    prompt = build_mistral_prompt(query, clauses)

    try:
        response = genai_model.generate_content(
            contents=[
                {"role": "user", "parts": [prompt]}
            ]
        )
        content = response.text.strip()

        # Handle JSON in markdown-style response
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        result = json.loads(content)
        return result

    except json.JSONDecodeError:
        return {
            "answer": "The document does not contain any clear or relevant clause to address the query. Please refer to the policy document directly or contact the insurer for further clarification."
        }
    except Exception as e:
        print(f"Error calling GenAI API from llm.py: {e}")
        return {
            "answer": "Error",
            "supporting_clause": "None",
            "explanation": f"Error while calling LLM API: {str(e)}"
        }
