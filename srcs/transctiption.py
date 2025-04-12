import queue

import langdetect
import whisper
from PyQt5.QtCore import QThread, pyqtSignal


class TranscriptionEngine:
    """Handles speech-to-text transcription using Whisper."""

    def __init__(self, model_name: str = "base"):
        """Initialize the transcription engine with the specified model."""
        self.model = None
        self.model_name = model_name

    def load_model(self):
        """Lazy-load the whisper model when needed."""
        if not self.model:
            try:
                self.model = whisper.load_model(self.model_name)
            except Exception as e:
                raise Exception(f"Failed to load Whisper model: {str(e)}")

    def correct_common_error(self, transcribed_text):
        errors = {
            "高速": "告诉",
            "到處": "告诉",
            "倒速": "告诉",
            "到速": "告诉",
            "倒数": "告诉",
            "成課": "乘客",
            "成课": "乘客",
            "成個": "乘客",
            "成刻": "乘客",
            "雨嬰": "语音",
            "余音": "语音",
            "王室": "模式",
            "捷克": "接客",
            "goita": "kereta",
        }
        for incorrect, correct in errors.items():
            transcribed_text = transcribed_text.replace(incorrect, correct)
        return transcribed_text

    def transcribe(self, audio_file: str) -> str:
        """Transcribe the audio file to text."""
        try:
            self.load_model()
            result = self.model.transcribe(audio_file, fp16=False)
            text = result.get("text", "").strip()
            return self.correct_common_error(text)
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")


class TranscriptionProcessor(QThread):
    """Processes audio files for transcription in a separate thread."""

    transcription_ready = pyqtSignal(str, str)  # (filepath, transcription text)
    transcription_error = pyqtSignal(str, str)  # (filepath, error message)

    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.transcription_engine = TranscriptionEngine()

    def add_file(self, path: str):
        """Add a file to the transcription queue."""
        self.queue.put(path)

    def stop(self):
        """Stop the transcription thread."""
        self.running = False

    def run(self):
        """Process files in the queue for transcription."""
        # Initialize the model in the thread
        try:
            self.transcription_engine.load_model()
        except Exception as e:
            self.transcription_error.emit("Model loading", str(e))
            self.running = False
            return

        while self.running:
            try:
                file_path = self.queue.get(timeout=0.5)
                text = self.transcription_engine.transcribe(file_path)
                self.transcription_ready.emit(file_path, text)
            except queue.Empty:
                continue
            except Exception as e:
                file_name = file_path if isinstance(file_path, str) else "Unknown file"
                self.transcription_error.emit(file_name, str(e))
