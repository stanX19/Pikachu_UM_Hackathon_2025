import asyncio
import os
import edge_tts

lang_map = {
    "en": "en-US-GuyNeural",
    "ms": "ms-MY-OsmanNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "ko": "zh-CN-XiaoxiaoNeural",
    "id": "ms-MY-OsmanNeural"
}

def synthesize_and_play(text, lang="en"):
    if not text.strip():
        print("[TTS] Skipped empty response.")
        return None

    print(f"[Edge-TTS] Speaking in {lang}: {text}")

    lines = text.strip().splitlines()
    content = ""
    for line in lines:
        if not line.strip().lower().startswith("(in"):
            content += line.strip() + " "

    content = content.strip()
    if not content:
        print("[TTS] No valid speech content found.")
        return None

    voice = lang_map.get(lang.lower(), "en-US-GuyNeural")
    output_file = "output_edge_tts.mp3"

    async def speak():
        communicate = edge_tts.Communicate(content, voice)
        await communicate.save(output_file)

    try:
        asyncio.run(speak())
        return output_file  # Return to Gradio
    except Exception as e:
        print(f"[Edge-TTS ERROR] {e}")
        return None
