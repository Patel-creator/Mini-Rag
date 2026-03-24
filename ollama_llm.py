import requests

def generate_answer_ollama(context, question):

    prompt = f"""
Answer only using the context.
If not found, say Not found.

Context:
{context}

Question:
{question}

Answer:
"""

    url = "http://localhost:11434/api/generate"

    data = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data)

    return response.json()["response"]