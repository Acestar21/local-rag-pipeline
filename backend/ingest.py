import pdfplumber
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

SAMPLE_PDF = Path(__file__).parent.parent/"sample/Kushal_Resume.pdf"

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
