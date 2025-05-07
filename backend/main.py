# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import traceback
import os
from fastapi import HTTPException

from src.detection_utils import detect_from_bytes
from src.music_gen       import generate_music

app = FastAPI(title="Emotion→Music API")

# allow your frontend (dev or prod) to talk here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount “static/” so generated .wav files are served:
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict_emotion")
async def predict_emotion_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")
    data = await file.read()
    emotion = detect_from_bytes(data)
    return {"emotion": emotion}

class MusicRequest(BaseModel):
    emotion: str
    genre:   str

@app.post("/generate_music")
def generate_music_endpoint(req: MusicRequest):
    try:
        out_path = generate_music(req.emotion, req.genre)
        filename = os.path.basename(out_path)
        return {"url": f"/static/{filename}"}
    except Exception as e:
        # Print full traceback to your Uvicorn console:
        traceback.print_exc()
        # Return the error message to the client (for dev only):
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
