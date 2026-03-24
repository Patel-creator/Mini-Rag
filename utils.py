import os

def load_documents(folder_path="data"):
    texts = []

    for file in os.listdir(folder_path):

        path = os.path.join(folder_path, file)

        # txt or md
        if file.endswith(".txt") or file.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    return texts

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks