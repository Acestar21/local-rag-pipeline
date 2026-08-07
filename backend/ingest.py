import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pdfplumber
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer, util

SAMPLE_PDF = Path(__file__).parent.parent / "sample/Kushal_Resume.pdf"

# INGEST
app = FastAPI()

print(SAMPLE_PDF)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    embeddings = [model.encode(chunk) for chunk in chunks]
    return embeddings


def create_database():
    conn = sqlite3.connect("document_embeddings.db")
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT,
                        content TEXT,
                        embedding BLOB
                    )""")
    cursor.execute("""DELETE FROM chunks""")  # clear on each run

    conn.commit()
    conn.close()


def insert_chunks(chunks, embeddings):
    conn = sqlite3.connect("document_embeddings.db")
    cursor = conn.cursor()

    for i in range(len(chunks)):
        content = chunks[i]
        embedding = embeddings[i].tobytes()  # Convert NumPy array to bytes

        cursor.execute(
            """INSERT INTO chunks (source, content, embedding)
                        VALUES (?, ?, ?)""",
            ("PDF Source", content, embedding),
        )

    conn.commit()
    conn.close()


def verify_row_count():
    conn = sqlite3.connect("document_embeddings.db")
    cursor = conn.cursor()

    # Get the row count
    cursor.execute("""SELECT COUNT(*) FROM chunks""")
    row_count = cursor.fetchone()[0]
    print(f"Number of rows in the table: {row_count}")

    conn.close()


# RETRIEVAL


def load_chunks_and_embeddings():
    conn = sqlite3.connect("document_embeddings.db")
    cursor = conn.cursor()

    cursor.execute("""
                    SELECT content,
                    embedding FROM chunks
        """)
    rows = cursor.fetchall()

    chunks = [row[0] for row in rows]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]

    conn.close()
    return chunks, embeddings


def embed_query(query):
    query_embedding = model.encode(query)
    return query_embedding


def compute_cosine_similarity(query_embedding, chunks, embeddings):
    similarities = [
        util.cos_sim(query_embedding, emb)[0][0].item() for emb in embeddings
    ]
    return chunks, similarities


def get_top_3_chunks(chunks, similarities):
    sorted_indices = np.argsort(similarities)[::-1][:3]
    return [chunks[idx] for idx in sorted_indices]


# LOCAL LLM


def join_context(chunks):
    return "".join(chunks)


def build_prompt(context, question):
    return f"Context: {context}\n\nQuestion: {question}\nAnswer only based on the context above."


def call_ollama_api(prompt, model_name="qwen2.5:7b"):
    url = "http://localhost:11434/api/generate"
    data = {"prompt": prompt, "model": model_name}
    response = requests.post(url, json=data, stream=True)

    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                yield chunk.get("response", "")
    else:
        yield f"Error: {response.status_code}"


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Not a PDF file")

    content = await file.read()

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        text = " ".join(page.extract_text() for page in pdf.pages)

    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    create_database()
    insert_chunks(chunks, embeddings)

    return {"message": "PDF ingested", "chunks": len(chunks)}


@app.post("/query")
async def query_user_question(question: str = Form(...)):
    chunks, embeddings = load_chunks_and_embeddings()
    _, similarities = compute_cosine_similarity(
        embed_query(question), chunks, embeddings
    )
    top_chunks = get_top_3_chunks(chunks, similarities)
    context = join_context(top_chunks)
    prompt = build_prompt(context, question)

    def generate():
        for chunk in call_ollama_api(prompt):
            yield f"data:{chunk}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
