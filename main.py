from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import os
import google.generativeai as genai
import fitz  # PyMuPDF

app = FastAPI(
    title="Plagia AI Backend",
    description="API for plagiarism, AI detection, and summarization.",
    version="0.1.0"
)

# Configure Gemini API Key from Render environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class AnalysisResult(BaseModel):
    summary: str
    ai_detection_score: float

# ---- Core Logic ----
async def analyze_text(text: str):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        # --- Summarization ---
        summary_prompt = "Summarize the following text in 3 sentences:\n\n" + text
        summary_response = await model.generate_content_async(summary_prompt)
        summary_text = summary_response.text

        # --- AI Detection Scoring ---
        ai_detection_prompt = (
            "Analyze the following text and return ONLY a number between 0 and 100 "
            "indicating how likely it is AI-generated "
            "(0 = definitely human, 100 = definitely AI):\n\n"
            + text
        )
        ai_detection_response = await model.generate_content_async(ai_detection_prompt)

        try:
            # Convert model response to float
            ai_score = float(ai_detection_response.text.strip())
        except:
            # fallback if Gemini gives something unexpected
            ai_score = 50.0  

        return summary_text, ai_score

    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error in analysis", -1.0

# ---- API Routes ----
@app.get("/")
def read_root():
    return {"message": "Hello from the Gemini-powered AI & Summary Backend!"}

@app.post("/api/analysis")
async def perform_analysis(file: UploadFile = File(...)):
    """
    Performs AI detection (0–100) and summarization on an uploaded text or PDF file.
    """
    try:
        content = await file.read()

        # Check if file is PDF
        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
        else:
            # Assume plain text for other files
            text = content.decode("utf-8")

        summary, ai_score = await analyze_text(text)

        return AnalysisResult(
            summary=summary,
            ai_detection_score=ai_score
        )
    except Exception as e:
        return {"error": str(e)}
