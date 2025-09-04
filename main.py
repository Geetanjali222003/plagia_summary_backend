from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import os
import google.generativeai as genai

app = FastAPI(
    title="Plagia AI Backend",
    description="API for plagiarism, AI detection, and summarization.",
)

# This will get your API key from the environment variables when deployed on Render
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

class AnalysisResult(BaseModel):
    summary: str
    ai_detection_score: float

async def analyze_text(text: str):
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Summarization Logic
        summary_prompt = "Summarize the following text in 3 sentences:\n\n" + text
        summary_response = await model.generate_content_async(summary_prompt)
        summary_text = summary_response.text

        # AI Detection Logic
        ai_detection_prompt = "Is the following text likely to be written by a human or an AI? Respond with only 'Human' or 'AI'.\n\n" + text
        ai_detection_response = await model.generate_content_async(ai_detection_prompt)
        
        # Simple score based on response
        if "AI" in ai_detection_response.text:
            ai_score = 1.0
        else:
            ai_score = 0.0

        return summary_text, ai_score
        
    except Exception as e:
        # Return a clear error if the API call fails
        print(f"Gemini API error: {e}")
        return "Error in analysis", -1.0

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Hello from the new Gemini-powered AI & Summary Backend!"}

@app.post("/api/analysis")
async def perform_analysis(file: UploadFile = File(...)):
    """
    Performs AI detection and summarization on an uploaded text file.
    """
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        summary, ai_score = await analyze_text(text)
        
        return AnalysisResult(
            summary=summary,
            ai_detection_score=ai_score
        )
    except Exception as e:
        return {"error": str(e)}