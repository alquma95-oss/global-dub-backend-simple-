import os
import uvicorn
import yt_dlp
import asyncio
import edge_tts
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ---------------------------
# LANGUAGE → VOICE MAPPING
# ---------------------------
VOICE_MAP = {
    "hindi": "hi-IN-SwaraNeural",
    "english": "en-US-JennyNeural",
    "arabic": "ar-EG-SalmaNeural",
    "japanese": "ja-JP-NanamiNeural",
    "korean": "ko-KR-SunHiNeural",
    "tamil": "ta-IN-PriyaNeural"
}

DEFAULT_VOICE = os.getenv("EDGE_TTS_VOICE", "hi-IN-SwaraNeural")


# ---------------------------
# REQUEST MODEL
# ---------------------------
class DubRequest(BaseModel):
    url: str
    language: str


# ---------------------------
# YOUTUBE AUDIO DOWNLOAD
# ---------------------------
def download_audio(url):
    try:
        filename = "input_audio.mp3"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": filename,
            "quiet": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return filename
    except Exception as e:
        print("Download Error:", e)
        return None


# ---------------------------
# WHISPER TRANSCRIPTION
# ---------------------------
def transcribe_audio(filepath):
    with open(filepath, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-tts",
            file=f
        )
    return transcript.text


# ---------------------------
# TRANSLATE USING GPT-4
# ---------------------------
def translate_text(text, target_lang):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Translate text to {target_lang}. Maintain original emotions and tone."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content


# ---------------------------
# GENERATE DUBBED AUDIO
# ---------------------------
async def synthesize_voice(text, lang):
    voice = VOICE_MAP.get(lang.lower(), DEFAULT_VOICE)
    output_file = "dubbed_audio.mp3"

    tts = edge_tts.Communicate(text, voice)
    await tts.save(output_file)

    return output_file


# ---------------------------
# MAIN PROCESSING API
# ---------------------------
@app.post("/dub")
async def dub_video(req: DubRequest):
    video_url = req.url
    target_language = req.language.lower()

    # 1) DOWNLOAD AUDIO
    audio_path = download_audio(video_url)
    if not audio_path:
        return {"error": "Failed to download video."}

    # 2) TRANSCRIBE
    original_text = transcribe_audio(audio_path)

    # 3) TRANSLATE
    translated_text = translate_text(original_text, target_language)

    # 4) TTS
    dubbed_audio = await synthesize_voice(translated_text, target_language)

    return FileResponse(
        dubbed_audio,
        media_type="audio/mpeg",
        filename="dubbed_audio.mp3"
    )


# ---------------------------
# ROOT CHECK
# ---------------------------
@app.get("/")
def home():
    return {"status": "Backend Running Successfully!"}


# Run local
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
