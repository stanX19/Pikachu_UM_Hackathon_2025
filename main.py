import sys
import os
import queue
import threading
import collections
import time
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
import webrtcvad
import whisper
from scipy.io.wavfile import write
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# Constants
SAVE_DIR = './data'
SAMPLE_RATE = 32000
FRAME_DURATION = 30  # ms, can only be 10, 20 or 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_TIMEOUT = 1.0  # seconds
VAD_AGGRESSIVENESS = 2  # 0-3, higher means more aggressive filtering


class AudioUtils:
    @staticmethod
    def ensure_directories():
        """Ensure all required directories exist."""
        os.makedirs(SAVE_DIR, exist_ok=True)

    @staticmethod
    def get_next_filename(file_index: int) -> str:
        """Generate a unique filename for the recording."""
        return os.path.join(SAVE_DIR, f'recorded_audio_{file_index}.wav')


class VoiceActivityDetector:
    """Handles voice activity detection using webrtcvad."""

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS):
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, audio_frame: bytes, sample_rate: int) -> bool:
        """Check if an audio frame contains speech."""
        try:
            return self.vad.is_speech(audio_frame, sample_rate)
        except Exception as e:
            print(f"VAD error: {e}")
            return False


class AudioRecorder(QThread):
    """Records audio when voice activity is detected and stops after silence."""

    finished = pyqtSignal(object)  # Signal emits either a filepath or an error

    def __init__(self, file_index: int):
        super().__init__()
        self.file_index = file_index
        self.vad = VoiceActivityDetector()
        self.buffer = collections.deque()
        self.running = True
        self.audio_queue = queue.Queue()

    def callback(self, indata, frames, time, status):
        """Callback for sounddevice InputStream."""
        if status:
            print(f"Stream status: {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def stop(self):
        """Stop the recording thread."""
        self.running = False

    def run(self):
        """Record audio with voice activity detection."""
        try:
            with sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=FRAME_SIZE,
                    dtype='int16',
                    channels=1,
                    callback=self.callback
            ):
                self._process_audio_stream()

        except Exception as e:
            self.finished.emit(Exception(f"Recording failed: {str(e)}"))

    def _process_audio_stream(self):
        """Process incoming audio data and detect speech segments."""
        silent_chunks = 0
        voiced_chunks = 0
        collected_frames = []
        max_silence_chunks = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION)

        speech_started = False
        speech_start_index = 0

        while self.running:
            try:
                frame = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

            if is_speech and not speech_started:
                speech_start_index = len(collected_frames)
                speech_started = True

            collected_frames.append(frame)

            if is_speech:
                silent_chunks = 0
                voiced_chunks += 1
            else:
                silent_chunks += 1

            # If we've detected speech and then sufficient silence, stop recording
            if voiced_chunks > 5 and silent_chunks > max_silence_chunks:
                break

        self._save_audio_if_speech_detected(collected_frames, speech_start_index, voiced_chunks)

    def _save_audio_if_speech_detected(self, collected_frames, speech_start_index, voiced_chunks):
        """Save the recorded audio if speech was detected."""
        if voiced_chunks <= 5:
            self.finished.emit("No speech detected")
            return

        try:
            # Trim beginning silence and convert to numpy array
            trimmed_frames = collected_frames[speech_start_index:]
            audio_data = b''.join(trimmed_frames)
            audio_np = np.frombuffer(audio_data, dtype='int16')

            # Save to file
            file_path = AudioUtils.get_next_filename(self.file_index)
            write(file_path, SAMPLE_RATE, audio_np)
            self.finished.emit(file_path)

        except Exception as e:
            self.finished.emit(Exception(f"Error saving audio: {str(e)}"))


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

    def transcribe(self, audio_file: str) -> str:
        """Transcribe the audio file to text."""
        try:
            self.load_model()
            result = self.model.transcribe(audio_file)
            return result.get("text", "").strip()
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")


class TranscriptionProcessor(QThread):
    """Processes audio files for transcription in a separate thread."""

    transcription_ready = pyqtSignal(str, str)  # (filepath, transcription text)
    transcription_error = pyqtSignal(str, str)  # (filepath, error message)

    def __init__(self, model_name: str = "base"):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.transcription_engine = TranscriptionEngine(model_name)

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


class AudioRecorderApp(QWidget):
    """Main application window for the voice-activated recorder."""

    def __init__(self):
        super().__init__()
        AudioUtils.ensure_directories()

        self.file_counter = 1
        self.recording = False
        self.recorder_thread = None

        self._setup_ui()
        self._setup_transcription_engine()

    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Voice-Activated Recorder")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Recording button
        self.record_button = QPushButton("Start Voice Recording")
        self.record_button.setStyleSheet("""
            QPushButton { 
                border-radius: 40px; 
                min-width: 80px; 
                min-height: 80px; 
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px;")

        # Transcription result label
        self.transcription_label = QLabel("Transcription will appear here")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setWordWrap(True)
        self.transcription_label.setStyleSheet("font-size: 12px; padding: 10px;")

        layout.addWidget(self.record_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.transcription_label)

        self.setLayout(layout)

    def _setup_transcription_engine(self):
        """Set up the transcription engine."""
        self.transcription_processor = TranscriptionProcessor()
        self.transcription_processor.transcription_ready.connect(self.on_transcription_ready)
        self.transcription_processor.transcription_error.connect(self.on_transcription_error)
        self.transcription_processor.start()

    def toggle_recording(self):
        """Toggle recording state on/off."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start voice recording."""
        self.recording = True
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("""
            QPushButton { 
                border-radius: 40px; 
                min-width: 80px; 
                min-height: 80px; 
                font-size: 16px;
                background-color: #f44336;
                color: white;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.status_label.setText("Listening for speech...")
        self.record_next()

    def stop_recording(self):
        """Stop voice recording."""
        self.recording = False
        self.record_button.setText("Start Voice Recording")
        self.record_button.setStyleSheet("""
            QPushButton { 
                border-radius: 40px; 
                min-width: 80px; 
                min-height: 80px; 
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.status_label.setText("Recording stopped")

        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop()

    def record_next(self):
        """Start the next recording session."""
        if not self.recording:
            return

        self.recorder_thread = AudioRecorder(self.file_counter)
        self.recorder_thread.finished.connect(self.on_recording_finished)
        self.recorder_thread.start()

    def on_recording_finished(self, result):
        """Handle completed recording."""
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Error", str(result))
            self.stop_recording()
        elif result == "No speech detected":
            self.status_label.setText("No speech detected, listening again...")
            if self.recording:
                self.record_next()
        else:
            # Recording successful
            self.status_label.setText(f"Processing: {os.path.basename(result)}")
            self.file_counter += 1
            self.transcription_processor.add_file(result)

            if self.recording:
                self.record_next()

    def on_transcription_ready(self, file_path, text):
        """Handle completed transcription."""
        file_name = os.path.basename(file_path)
        self.status_label.setText(f"Transcribed: {file_name}")
        self.transcription_label.setText(text)
        print(f"Transcribed [{file_name}]: {text}")

    def on_transcription_error(self, file_path, error_msg):
        """Handle transcription error."""
        print(f"Transcription error for {file_path}: {error_msg}")
        self.status_label.setText(f"Transcription error: {os.path.basename(file_path)}")

    def closeEvent(self, event):
        """Clean up resources when closing the application."""
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop()

        if self.transcription_processor:
            self.transcription_processor.stop()
            self.transcription_processor.wait()

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioRecorderApp()
    window.show()
    sys.exit(app.exec_())