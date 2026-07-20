import pdfplumber
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import json
import requests

SAMPLE_PDF = Path(__file__).parent.parent/"sample/Kushal_Resume.pdf"

# INGEST

print(SAMPLE_PDF)
with pdfplumber.open(SAMPLE_PDF) as pdf:
    first_page = pdf.pages[0].extract_text()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    embeddings = [model.encode(chunk) for chunk in chunks]
    return embeddings

CHUNKS = chunk_text(first_page)
EMBEDDING = embed_chunks(CHUNKS)


def create_database():
    conn = sqlite3.connect('document_embeddings.db')
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT,
                        content TEXT,
                        embedding BLOB
                    )''')
    cursor.execute('''DELETE FROM chunks''')  # clear on each run
    
    conn.commit()
    conn.close()

def insert_chunks(chunks, embeddings):
    conn = sqlite3.connect('document_embeddings.db')
    cursor = conn.cursor()
    
    for i in range(len(chunks)):
        content = chunks[i]
        embedding = embeddings[i].tobytes()  # Convert NumPy array to bytes
        
        cursor.execute('''INSERT INTO chunks (source, content, embedding) 
                        VALUES (?, ?, ?)''', ('PDF Source', content, embedding))
    
    conn.commit()
    conn.close()

def verify_row_count():
    conn = sqlite3.connect('document_embeddings.db')
    cursor = conn.cursor()
    
    # Get the row count
    cursor.execute('''SELECT COUNT(*) FROM chunks''')
    row_count = cursor.fetchone()[0]
    print(f"Number of rows in the table: {row_count}")
    
    conn.close()


create_database()

insert_chunks(CHUNKS, EMBEDDING)

verify_row_count()

# RETRIEVAL 

def load_chunks_and_embeddings():
    conn = sqlite3.connect('document_embeddings.db')
    cursor = conn.cursor()

    cursor.execute('''
                    SELECT content,
                    embedding FROM chunks
        ''')
    rows = cursor.fetchall()

    chunks = [row[0] for row in rows]
    embeddings = [np.frombuffer(row[1] , dtype = np.float32) for row in rows ]

    conn.close()
    return chunks, embeddings

def embed_query(query):
    query_embedding = model.encode(query)
    return query_embedding

def compute_cosine_similarity(query_embedding, chunks, embeddings):
    similarities = [util.cos_sim(query_embedding, emb)[0][0].item() for emb in embeddings]
    return chunks, similarities

def print_top_3_relevant_chunks(chunks, similarities):
    sorted_indices = np.argsort(similarities)[::-1][:3]
    
    for idx in sorted_indices:
        print(f"Chunk: {chunks[idx]}")
        print(f"Similarity: {similarities[idx]:.4f}")

# LOCAL LLM

def join_context(chunks):
    return "".join(chunks)

def build_prompt(context):
    prompt = f"Context: {context}\n\n Question: What are your skills?"
    return prompt

def call_ollama_api(prompt , model_name = "qwen2.5:7b"):
    url = "http://localhost:11434/api/generate"

    data = {
        "prompt": prompt,
        "model": model_name
    }

    response = requests.post(url , json = data, stream = True)

    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                print(data.get("response", ""), end="", flush=True)

    else :
        print(f"Failed to fetch data from Ollama API. Status code: {response.status_code}")

chunks, embeddings = load_chunks_and_embeddings()

query = "What are your skills?"
query_embedding = embed_query(query)

chunks, similarities = compute_cosine_similarity(query_embedding, chunks, embeddings)
print_top_3_relevant_chunks(chunks, similarities)

top_3_chunks = [chunks[i] for i in np.argsort(similarities)[::-1][:3]]
context = join_context(top_3_chunks)
prompt = build_prompt(context)
        
call_ollama_api(prompt, model_name="qwen2.5:7b")

