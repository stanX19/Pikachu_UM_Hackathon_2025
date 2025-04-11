import gradio as gr
from voice_pipeline import process_audio_pipeline
from llm_handler import query_llm
from tts_engine import synthesize_and_play

def handle_interaction():
    transcription, lang = process_audio_pipeline()
    print(f"[You said]: {transcription} | [Detected]: {lang}")

    if transcription.strip().lower() in ["no speech detected", ""]:
        return transcription, "No response generated.", None

    response = query_llm(transcription, language=lang)
    audio_path = synthesize_and_play(response, lang=lang)
    return transcription, response, audio_path

def launch_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("### 🎙️ Grab DAX Voice Assistant Prototype")
        record_btn = gr.Button("Start Talking")
        transcription_output = gr.Textbox(label="You Said")
        response_output = gr.Textbox(label="AI Response")
        audio_output = gr.Audio(label="Voice Reply", autoplay=True)

        record_btn.click(fn=handle_interaction, inputs=[], outputs=[transcription_output, response_output, audio_output])

    demo.launch(server_port=7865, show_error=True)
