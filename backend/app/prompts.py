from transformers import AutoTokenizer

# Load tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Template for single question prompt
MISTRAL_SYSTEM_PROMPT_TEMPLATE = """
You are a helpful insurance assistant. Your task is to read the given policy clauses and answer the user's question clearly and naturally, using only the information in the clauses.

Instructions:
- Start your answer with "Yes" or "No", based only on what's explicitly stated.
- Use simple, natural language that anyone can understand.
- Do NOT guess, assume, or include outside knowledge.
- Do NOT mention clause numbers, section names, or formatting.
- Be specific, complete, and keep the answer under 4 lines (ideally <25 words).
- Include key details such as conditions, limits, waiting periods, or exclusions.
- If the answer is partially implied, infer it cautiously and explain using clause text.
- If exact phrases are missing, use semantic meaning or synonyms to match the question with clause content. Rely on reasoning if partial matches exist.

Output format:
{
  "answer": "<A full-sentence, clear answer starting with 'Yes' or 'No', using only clause content>"
}

User Question:
{query}

Relevant Policy Clauses:
{clauses}

Respond with only the raw JSON (no markdown, no extra text, no backticks).
""".strip()

# 🔹 Utility: Trim clauses by token limit
def _trim_clauses(clauses: list, max_tokens: int) -> str:
    trimmed = []
    total_tokens = 0

    for clause_obj in clauses:
        clause = clause_obj.get("clause", "").strip()
        tokens = len(tokenizer.tokenize(clause))
        if total_tokens + tokens > max_tokens:
            break
        trimmed.append(clause)
        total_tokens += tokens

    return "\n\n".join(trimmed)

# 🔹 Single-question prompt builder
def build_mistral_prompt(query: str, clauses: list, max_tokens: int = 1800) -> str:
    clause_text = _trim_clauses(clauses, max_tokens)
    return MISTRAL_SYSTEM_PROMPT_TEMPLATE.format(
        query=query.strip(),
        clauses=clause_text
    )

# 🔹 Multi-question batch prompt builder
def build_batch_prompt(questions: list, clause_map: dict, max_tokens: int = 1800) -> str:
    all_clauses = set()
    for clause_list in clause_map.values():
        all_clauses.update(clause['clause'] for clause in clause_list)

    clause_text = _trim_clauses([{'clause': c} for c in all_clauses], max_tokens)

    question_block = "\n".join([
        f"Q{i+1}: {q.strip()}"
        for i, q in enumerate(questions)
    ])

    return f"""
You are an expert insurance assistant. Read the policy clauses below and answer the user's questions strictly using only the clause content.

Policy Clauses:
{clause_text}

User Questions:
{question_block}

Answer in this JSON format:
{{
  "Q1": "answer to question 1",
  "Q2": "answer to question 2",
  ...
}}

Instructions:
- Be concise and factual (max 25 words per answer).
- Start each answer with "Yes" or "No" based on what’s stated in the clause.
- Use reasoning based on synonyms and meanings, not just exact phrase match.
- Do NOT guess, assume, or include outside knowledge.
- If the clauses do not contain relevant information, respond with: "No, the policy does not cover this."
- Respond ONLY with the raw JSON (no markdown or text).
""".strip()
