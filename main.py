import os
import uuid
import yt_dlp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import edge_tts

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DubRequest(BaseModel):
    video_url: str
    target_language: str


@app.post("/dub")
async def dub_video(req: DubRequest):
    url = req.video_url
    lang = req.target_language.lower()

    audio_filename = f"input_{uuid.uuid4()}.mp3"
    output_audio = f"output_{uuid.uuid4()}.mp3"

    # 1️⃣ DOWNLOAD YOUTUBE AUDIO (NO COOKIES NEEDED)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_filename,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        return {"error": f"Audio download failed: {str(e)}"}

    # 2️⃣ TRANSCRIBE WITH EMOTIONS (Whisper v3 large)
    try:
        transcript = client.audio.transcriptions.create(
            file=open(audio_filename, "rb"),
            model="gpt-4o-mini-transcribe",
            response_format="verbose_json"
        )
        original_text = transcript.text
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}

    # 3️⃣ TRANSLATE WITH EMOTION PRESERVATION
    try:
        translation = client.responses.create(
            model="gpt-4o-mini",
            input=f"""
            Translate to {lang}.
            Keep ALL emotions exactly same (happy/sad/excited/angry).
            Text: {original_text}
            """
        )
        translated = translation.output[0].content[0].text
    except Exception as e:
        return {"error": f"Translation failed: {str(e)}"}

    # 4️⃣ EMOTIONAL TTS
    VOICES = {
        "english": "en-US-JennyNeural",
        "hindi": "hi-IN-SwaraNeural",
        "japanese": "ja-JP-NanamiNeural",
        "korean": "ko-KR-SunHiNeural",
        "arabic": "ar-EG-SalmaNeural",
    }

    if lang not in VOICES:
        return {"error": "Language not supported"}

    try:
        tts = edge_tts.Communicate(translated, VOICES[lang])
        await tts.save(output_audio)
    except Exception as e:
        return {"error": f"TTS failed: {str(e)}"}

    return {
        "status": "success",
        "audio_url": f"https://global-dub-backend.onrender.com/{output_audio}"
    }
