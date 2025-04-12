import asyncio
import threading

import edge_tts
import os
import uuid
import time
import pygame

lang_map = {
    "en": "en-US-GuyNeural",
    "ms": "ms-MY-OsmanNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "zh-tw": "zh-CN-XiaoxiaoNeural",
    "ko": "zh-CN-XiaoxiaoNeural",
    "id": "ms-MY-OsmanNeural"
}
pygame.mixer.init()

def play_audio_in_background(file_path):
    """Play audio in a separate thread and return immediately"""

    def _play_audio():
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing audio: {e}")

    # Start audio in a separate thread
    audio_thread = threading.Thread(target=_play_audio)
    audio_thread.daemon = True  # Thread will close when main program exits
    audio_thread.start()
    return audio_thread


def synthesize_and_play(text, lang="en"):
    if not text.strip():
        return None
    print(f"[Edge-TTS] Speaking in {lang}: {text}")

    lines = text.strip().splitlines()
    content = " ".join(line.strip() for line in lines if not line.strip().lower().startswith("(in"))
    content = content.strip()
    if not content:
        return None

    voice = lang_map.get(lang.lower(), "en-US-GuyNeural")

    # Create a unique filename with absolute path
    data_dir = os.path.abspath("data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, f"output_{uuid.uuid4().hex}.mp3")

    async def generate_speech():
        communicate = edge_tts.Communicate(content, voice)
        await communicate.save(output_path)

    try:
        # Generate speech
        asyncio.run(generate_speech())

        # Verify file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:

            # Start playing audio in background and return immediately
            play_audio_in_background(output_path)

            return True
        else:
            return False
    except Exception as e:
        print(f"[Edge-TTS ERROR] {e}")
        return None