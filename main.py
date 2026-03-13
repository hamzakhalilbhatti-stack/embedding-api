from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class TextInput(BaseModel):
    text: str

@app.post("/embed")
def embed_text(input: TextInput):
    vector = model.encode(input.text).tolist()
    return {"embedding": vector}