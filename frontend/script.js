async function sendQuery() {
  const query = document.getElementById("query").value;
  const res = await fetch("http://localhost:8000/hackrx/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query })
  });
  const data = await res.json();
  document.getElementById("response").textContent = JSON.stringify(data, null, 2);
}

//openai.api_base = "https://openrouter.ai/api/v1"
//     openai.api_key = "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" 