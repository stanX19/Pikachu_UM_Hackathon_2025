import whisper

model = whisper.load_model("base")

def transcribe_audio(audio):
    return model.transcribe(audio)['text']
