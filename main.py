from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import google.generativeai as genai
import fitz  # PyMuPDF
import os

# Configure Gemini API with your key (set this in Render Environment Variables)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use Gemini Pro instead of Flash
model = genai.GenerativeModel("gemini-1.5-pro-latest")

app = FastAPI(
    title="Plagia AI Backend",
    version="0.1.0",
    description="API for plagiarism, AI detection, and summarization."
)

@app.get("/")
def read_root():
    return {"message": "Plagia AI Backend is running with Gemini Pro 🚀"}

# Extract text from uploaded PDF or text file
def extract_text(file: UploadFile) -> str:
    text = ""
    if file.filename.endswith(".pdf"):
        pdf_document = fitz.open(stream=file.file.read(), filetype="pdf")
        for page in pdf_document:
            text += page.get_text("text")
    else:
        text = file.file.read().decode("utf-8", errors="ignore")
    return text.strip()

@app.post("/api/analysis")
async def perform_analysis(file: UploadFile = File(...)):
    try:
        text = extract_text(file)

        if not text:
            return JSONResponse(content={"summary": "No text found", "ai_detection_score": -1}, status_code=200)

        # Summarization with Gemini Pro
        summary_prompt = f"Summarize the following text in 3-4 sentences:\n\n{text}"
        summary_response = model.generate_content(summary_prompt)
        summary = summary_response.text if summary_response and summary_response.text else "Error generating summary"

        # AI Detection with Gemini Pro
        detect_prompt = f"Analyze the following text and return a number between 0 (human-written) and 100 (completely AI-generated):\n\n{text}"
        detect_response = model.generate_content(detect_prompt)
        score = -1
        if detect_response and detect_response.text:
            try:
                score = int("".join(filter(str.isdigit, detect_response.text)))
                score = max(0, min(100, score))  # clamp 0–100
            except:
                score = -1

        return JSONResponse(content={
            "summary": summary,
            "ai_detection_score": score
        }, status_code=200)

    except Exception as e:
        print(f"Error in analysis: {e}")
        return JSONResponse(content={"summary": "Error in analysis", "ai_detection_score": -1}, status_code=200)
