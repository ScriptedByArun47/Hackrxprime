import ollama

MISTRAL_SYSTEM_PROMPT = "You are a legal assistant. Answer clearly in one sentence."

def build_prompt(query, clauses):
    return f"User Query: {query}\n\nRelevant Clauses:\n" + "\n\n".join(clauses)

query = "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
clauses = ["A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."]

prompt = build_prompt(query, clauses)

print("🟡 Prompt:\n", prompt)

response = ollama.chat(
    model="mistral",
    messages=[
        {"role": "system", "content": MISTRAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
)

print("🟢 LLM Response:\n", response['message']['content'])
