from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_engine import siapkan_ai_fisika

app = FastAPI()

# Izinkan HTML Anda mengakses Python ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI Halliday Anda
chatbot_halliday = siapkan_ai_fisika("../data/buku halliday.pdf")

class ChatRequest(BaseModel):
    message: str
    topic: str

@app.post("/ask")
async def ask_physics(data: ChatRequest):
    # Gabungkan topik ke dalam pertanyaan agar AI lebih fokus
    input_text = f"Topik: {data.topic}. Pertanyaan: {data.message}"
    response = chatbot_halliday.invoke({"input": input_text})
    return {"answer": response["answer"]}

@app.post("/generate-quiz")
async def make_quiz(data: ChatRequest):
    prompt_kuis = f"Buatlah 3 soal pilihan ganda tentang {data.topic} berdasarkan buku Halliday. Sertakan kunci jawabannya di akhir."
    response = chatbot_halliday.invoke({"input": prompt_kuis})
    return {"quiz": response["answer"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)