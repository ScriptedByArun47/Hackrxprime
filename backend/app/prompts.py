MISTRAL_SYSTEM_PROMPT_TEMPLATE = """
You are an expert insurance assistant. Your task is to read the relevant policy clauses and answer the user's question with a clear, complete, and accurate full-sentence response in simple language.

Instructions:
- ONLY use the information explicitly provided in the policy clauses.
- Do NOT assume, guess, or include outside knowledge.
- Do NOT mention clause numbers, section names, or document formatting.
- Your answer must be factual, specific, and based only on the content of the clauses(under 25 words).
- Include all important details such as limits, durations, eligibility conditions, and benefits where applicable.

Output format:
{{
  "answer": "<One complete and factual sentence derived strictly from the given clauses>"
}}

User Question:
{query}

Relevant Policy Clauses:
{clauses}

Respond with only the raw JSON (no markdown or formatting).
"""

from transformers import AutoTokenizer

# Load tokenizer only once
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def build_mistral_prompt(query: str, clauses: list, max_tokens: int = 1500) -> str:
    trimmed_clauses = []
    token_count = 0

    for clause_obj in clauses:
        clause = clause_obj["clause"].strip()
        tokens = len(tokenizer.tokenize(clause))
        if token_count + tokens > max_tokens:
            break
        trimmed_clauses.append(clause)
        token_count += tokens

    clause_text = "\n\n".join(trimmed_clauses)
    return MISTRAL_SYSTEM_PROMPT_TEMPLATE.format(query=query.strip(), clauses=clause_text)


# ✅ Batch prompt builder for answering multiple questions at once
def build_batch_prompt(questions: list, clauses: list, max_tokens: int = 1800) -> str:
    trimmed_clauses = []
    token_count = 0

    for clause_obj in clauses:
        clause = clause_obj["clause"].strip()
        tokens = len(tokenizer.tokenize(clause))
        if token_count + tokens > max_tokens:
            break
        trimmed_clauses.append(clause)
        token_count += tokens

    clause_text = "\n\n".join(trimmed_clauses)
    numbered_questions = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(questions)])

    return f"""You are an expert insurance assistant. Read the following policy clauses and answer all user questions based strictly on the clauses.

Policy Clauses:
{clause_text}

User Questions:
{numbered_questions}

Answer each question in JSON format like:
{{
  "Q1": "answer to question 1",
  "Q2": "answer to question 2",
  ...
}}

Only use clause content. Do not guess or assume. Concise answers(under 25 words). Answer each question clearly and concisely. Respond only with the raw JSON (no markdown or formatting).
"""