import sounddevice as sd
import numpy as np
import noisereduce as nr
import whisper
from langdetect import detect

model = whisper.load_model("small")

def record_audio(duration=5, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def reduce_noise(audio, fs=16000):
    return nr.reduce_noise(y=audio, sr=fs)

def transcribe(audio, fs=16000):
    audio_float32 = audio.astype(np.float32)
    return model.transcribe(audio_float32, fp16=False)['text']

def detect_language(text):
    return detect(text)

def process_audio_pipeline():
    audio = record_audio()
    clean_audio = reduce_noise(audio)
    transcription = transcribe(clean_audio)

    if not transcription.strip():  # Empty or whitespace
        print("Transcription is empty.")
        return "No speech detected", "en"

    language = detect_language(transcription)
    return transcription, language
