from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import os
import google.generativeai as genai
import fitz  # PyMuPDF

# Initialize FastAPI app
app = FastAPI(
    title="Plagia AI Backend",
    description="API for plagiarism, AI detection, and summarization.",
)

# Configure Gemini API key from environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Response model
class AnalysisResult(BaseModel):
    summary: str
    ai_detection_score: float

# --- Core Analysis Function ---
async def analyze_text(text: str):
    try:
        # Use supported Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Summarization
        summary_prompt = "Summarize the following text in 3 sentences:\n\n" + text
        summary_response = model.generate_content(summary_prompt)
        summary_text = summary_response.text.strip()

        # AI Detection
        ai_detection_prompt = (
            "Is the following text likely to be written by a human or an AI? "
            "Respond with only 'Human' or 'AI'.\n\n" + text
        )
        ai_detection_response = model.generate_content(ai_detection_prompt)
        ai_detection_result = ai_detection_response.text.strip()

        if "AI" in ai_detection_result:
            ai_score = 1.0
        else:
            ai_score = 0.0

        return summary_text, ai_score

    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error in analysis", -1.0


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Hello from the Gemini-powered AI & Summary Backend!"}

@app.post("/api/analysis")
async def perform_analysis(file: UploadFile = File(...)):
    """
    Performs AI detection and summarization on an uploaded text or PDF file.
    """
    try:
        content = await file.read()
        
        # If PDF, extract text with PyMuPDF
        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
        else:
            # Otherwise, treat as plain text
            text = content.decode("utf-8")
        
        summary, ai_score = await analyze_text(text)
        
        return AnalysisResult(
            summary=summary,
            ai_detection_score=ai_score
        )
    except Exception as e:
        return {"error": str(e)}
