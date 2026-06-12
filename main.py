
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import File, UploadFile
import os
import csv
import io
from pathlib import Path
from pydantic import BaseModel
from llm_provider import LLMProvider
from mangum import Mangum
print("test")

app = FastAPI()
handler = Mangum(app)

TEMPLATES_DIR = Path(__file__).parent / "templates"

@app.get("/", response_class=HTMLResponse)
def html():
    html_file = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))

async def get_file_content(file: UploadFile):
    contents = await file.read()
    text_str = contents.decode("utf-8", errors="replace")
    return text_str


CHUNK_SIZE = 1024 * 1024  # 1MB
OVERLAP_SIZE = 100  # 100 bytes
#define from openrouter
def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP_SIZE
    return chunks


def create_embeddings(text):
    try:
        llm_provider = LLMProvider("openrouter")
        embedding = llm_provider.create_embedding(text)
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

# store chunks and their embeddings
stored_chunks = []
stored_embeddings = []
stored_text = ""
stored_user_ids = []
user_id_rows = {}  # user_id -> list of row texts


# Cosine Similarity Algorithm:
# similarity = (A · B) / (||A|| * ||B||)
# where A · B = sum of element-wise products (dot product)
# ||A|| = sqrt(sum of squares of A) (Euclidean norm)
# Returns a value between -1 and 1, where 1 = identical, 0 = unrelated
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar_chunks(query_embedding, top_k=3):
    scored = []
    for i, emb in enumerate(stored_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scored.append((score, i))
    scored.sort(reverse=True)
    return scored[:top_k]
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global stored_text
    text = await get_file_content(file)
    stored_text = text
    chunks = chunk_text(text)
    embeddings = [create_embeddings(chunk) for chunk in chunks]
    stored_chunks.clear()
    stored_embeddings.clear()
    stored_chunks.extend(chunks)
    stored_embeddings.extend(embeddings)

    # extract user_ids from CSV
    stored_user_ids.clear()
    user_id_rows.clear()
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            uid = row.get("user_id", "").strip()
            if uid:
                if uid not in user_id_rows:
                    user_id_rows[uid] = []
                user_id_rows[uid].append(",".join(row.values()))
        stored_user_ids.extend(sorted(user_id_rows.keys()))
    except Exception:
        pass  # not a CSV or no user_id column

    return {"text": text, "chunks": chunks, "embeddings": embeddings, "user_ids": stored_user_ids}


@app.get("/users")
def get_users():
    return {"user_ids": stored_user_ids}

class ChatRequest(BaseModel):
    message: str
    user_id: str = ""

#this should take message:'message' from post request
@app.post("/chat")
def chat(req: ChatRequest):
    if not stored_embeddings:
        return {"reply": "Please upload a file first."}

    llm_provider = LLMProvider("openrouter")
    # embed the user's query
    query_embedding = llm_provider.create_embedding(req.message)
    # find most similar chunks (cosine similarity)
    top_matches = find_similar_chunks(query_embedding, top_k=3)
    # return the matching chunks
    results = []
    for score, idx in top_matches:
        results.append({
            "chunk_index": idx,
            "score": round(score, 4),
            "text": stored_chunks[idx][:500]  # first 500 chars preview
        })
    # add user-specific context if user_id is selected
    details_parts = [f"Chunk {r['chunk_index']} (score: {r['score']}): {r['text']}" for r in results]
    if req.user_id and req.user_id in user_id_rows:
        user_context = "\n".join(user_id_rows[req.user_id])
        details_parts.insert(0, f"Filtered data for user {req.user_id}:\n{user_context}")
    response = LLMProvider("openrouter").chat(req.message, details="\n".join(details_parts))
    return {"reply": response, "matches": results}

