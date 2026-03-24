import os
import PyPDF2

def load_documents(folder_path="data"):
    texts = []

    for file in os.listdir(folder_path):

        path = os.path.join(folder_path, file)

        # txt or md
        if file.endswith(".txt") or file.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        # pdf
        elif file.endswith(".pdf"):
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                    if text.strip():
                        texts.append(text)
            except Exception as e:
                print(f"Error reading {file}: {e}")

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