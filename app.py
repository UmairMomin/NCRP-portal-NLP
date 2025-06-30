#run: uvicorn app:app --reload
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import spacy
from sklearn.pipeline import Pipeline
import joblib
import os
from uuid import uuid4
from core import preprocess_text, train_model, create_pdf_report

# Google Gemini Imports
import google.generativeai as genai
from dotenv import load_dotenv

# Load env vars
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === FastAPI App ===
app = FastAPI()

# CORS config for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load/Train ML Model ===
MODEL_PATH = "cybercrime_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    df = pd.read_csv("complete_complaints_dataset.csv")
    model = train_model(df)
    joblib.dump(model, MODEL_PATH)

# === PDF Generation Endpoint ===
class ComplaintRequest(BaseModel):
    complaint: str

@app.post("/generate-pdf")
async def generate_pdf(req: ComplaintRequest):
    try:
        complaint_text = req.complaint
        processed = preprocess_text(complaint_text)
        category = model.predict([processed])[0]
        sub_category = "General"  # Placeholder

        filename = f"reports/{uuid4().hex}.pdf"
        os.makedirs("reports", exist_ok=True)
        create_pdf_report(complaint_text, category, sub_category, filename)

        return {"pdf_url": f"/download/{os.path.basename(filename)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    file_path = f"reports/{filename}"
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='application/pdf')
    raise HTTPException(status_code=404, detail="File not found")

# === Gemini Chatbot ===
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

chat_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="The NCRP Chatbot will assist users in filing cybercrime reports by guiding them step-by-step and using NLP to map their inputs to relevant cybercrime categories and laws..."
)

chat_session = chat_model.start_chat(history=[])
session_active = True

@app.post("/chat")
async def get_chat_response(msg: str = Form(...)):
    global session_active
    if not session_active:
        return JSONResponse(content={"response": "Close the chat and restart."})

    if msg.lower() in ["exit", "quit", "thanks for help"]:
        session_active = False
        return JSONResponse(content={"response": "Goodbye! Chat session has ended."})

    try:
        response = chat_session.send_message(msg)
        model_response = response.text.replace("**", "").replace("*", "")

        chat_session.history.append({"role": "user", "parts": [msg]})
        chat_session.history.append({"role": "model", "parts": [model_response]})

        return JSONResponse(content={"response": model_response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"response": str(e)})

