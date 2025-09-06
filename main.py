from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline
import fitz  # PyMuPDF

app = FastAPI()

# Load detector once at startup
detector = pipeline("text-classification", model="roberta-base")

@app.post("/detect")
async def detect(file: UploadFile = File(None), text: str = Form(None)):
    content = ""

    if text:
        content = text
    elif file and file.filename.endswith(".pdf"):
        file_bytes = await file.read()
        with open("temp.pdf", "wb") as f:
            f.write(file_bytes)
        with fitz.open("temp.pdf") as doc:
            content = "".join(page.get_text() for page in doc)

    if not content.strip():
        return {"error": "No text provided"}

    results = detector(content[:512])  # limit input length for performance
    return {"result": results, "input_snippet": content[:200]}
