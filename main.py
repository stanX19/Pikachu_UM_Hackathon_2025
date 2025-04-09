import sys
import os
import queue
import threading
import collections
import time
from typing import Optional, Callable, List, Tuple

import numpy as np
import sounddevice as sd
import webrtcvad
import whisper
from scipy.io.wavfile import write
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox,
                             QLabel, QStackedLayout, QHBoxLayout, QSizePolicy, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QIcon

# Constants
SAVE_DIR = './data'
SAMPLE_RATE = 32000
FRAME_DURATION = 30  # ms, can only be 10, 20 or 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_TIMEOUT = 1.0  # seconds
POST_SPEECH_BUFFER = 0.5  # seconds to keep after speech ends
VAD_AGGRESSIVENESS = 3  # 0-3, higher means more aggressive filtering

# Voice command constants
COMMAND_TIMEOUT = 5  # seconds to listen for a command
COMMAND_VOCABULARY = {
    "start recording": "start_recording",
    "stop recording": "stop_recording",
    "switch to normal": "switch_to_normal",
    "switch to voice": "switch_to_voice",
    "exit": "exit_app"
}


class AudioUtils:
    @staticmethod
    def ensure_directories():
        """Ensure all required directories exist."""
        os.makedirs(SAVE_DIR, exist_ok=True)

    @staticmethod
    def get_next_filename(file_index: int) -> str:
        """Generate a unique filename for the recording."""
        return os.path.join(SAVE_DIR, f'recorded_audio_{file_index}.wav')

    @staticmethod
    def trim_silence(audio_data: np.ndarray, is_speech_frames: List[bool]) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        if not any(is_speech_frames):
            return audio_data  # Return original if no speech detected

        # Find first speech frame
        start_idx = 0
        while start_idx < len(is_speech_frames) and not is_speech_frames[start_idx]:
            start_idx += 1

        # Find last speech frame
        end_idx = len(is_speech_frames) - 1
        while end_idx >= 0 and not is_speech_frames[end_idx]:
            end_idx -= 1

        # Add buffer frames after speech (but don't exceed array bounds)
        buffer_frames = int(POST_SPEECH_BUFFER * 1000 / FRAME_DURATION)
        end_idx = min(len(is_speech_frames) - 1, end_idx + buffer_frames)

        # Calculate frame indices to sample indices
        start_sample = start_idx * FRAME_SIZE
        end_sample = min(len(audio_data), (end_idx + 1) * FRAME_SIZE)

        return audio_data[start_sample:end_sample]


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
        is_speech_frames = []  # Track which frames contain speech
        max_silence_chunks = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION)

        speech_started = False

        while self.running:
            try:
                frame = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            is_speech_frames.append(is_speech)

            if is_speech and not speech_started:
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

        self._save_audio_if_speech_detected(collected_frames, is_speech_frames, voiced_chunks)

    def _save_audio_if_speech_detected(self, collected_frames, is_speech_frames, voiced_chunks):
        """Save the recorded audio if speech was detected, trimming silence from both ends."""
        if voiced_chunks <= 5:
            self.finished.emit("No speech detected")
            return

        try:
            # Convert to numpy array
            audio_data = b''.join(collected_frames)
            audio_np = np.frombuffer(audio_data, dtype='int16')

            # Trim silence from both ends
            trimmed_audio = AudioUtils.trim_silence(audio_np, is_speech_frames)

            # Save to file
            file_path = AudioUtils.get_next_filename(self.file_index)
            write(file_path, SAMPLE_RATE, trimmed_audio)
            self.finished.emit(file_path)

        except Exception as e:
            self.finished.emit(Exception(f"Error saving audio: {str(e)}"))


class CommandListener(QThread):
    """Listens for voice commands in voice control mode."""

    command_detected = pyqtSignal(str)  # Emits command name when detected
    listening_status = pyqtSignal(str)  # Updates listening status

    def __init__(self, transcription_engine):
        super().__init__()
        self.running = True
        self.transcription_engine = transcription_engine
        self.vad = VoiceActivityDetector()
        self.audio_queue = queue.Queue()

    def callback(self, indata, frames, time, status):
        """Callback for sounddevice InputStream."""
        if status:
            print(f"Stream status: {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def stop(self):
        """Stop the command listener."""
        self.running = False

    def run(self):
        """Listen for voice commands."""
        try:
            with sd.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=FRAME_SIZE,
                    dtype='int16',
                    channels=1,
                    callback=self.callback
            ):
                while self.running:
                    self._listen_for_command()

        except Exception as e:
            print(f"Command listening failed: {str(e)}")

    def _listen_for_command(self):
        """Listen for a single command."""
        self.listening_status.emit("Listening for command...")

        silent_chunks = 0
        voiced_chunks = 0
        collected_frames = []
        is_speech_frames = []
        max_silence_chunks = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION)
        max_command_chunks = int(COMMAND_TIMEOUT * 1000 / FRAME_DURATION)
        total_chunks = 0

        speech_started = False

        # Clear the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        while self.running:
            try:
                frame = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
            is_speech_frames.append(is_speech)
            collected_frames.append(frame)
            total_chunks += 1

            if is_speech:
                if not speech_started:
                    self.listening_status.emit("Speech detected, processing...")
                    speech_started = True
                silent_chunks = 0
                voiced_chunks += 1
            else:
                silent_chunks += 1

            # If we've detected speech and then sufficient silence, process command
            if voiced_chunks > 5 and silent_chunks > max_silence_chunks:
                break

            # If we've been listening too long, time out
            if total_chunks > max_command_chunks:
                if voiced_chunks <= 5:
                    self.listening_status.emit("No command detected")
                    return
                break

        if voiced_chunks <= 5:
            self.listening_status.emit("No command detected")
            return

        try:
            # Convert to numpy array and trim silence
            audio_data = b''.join(collected_frames)
            audio_np = np.frombuffer(audio_data, dtype='int16')
            trimmed_audio = AudioUtils.trim_silence(audio_np, is_speech_frames)

            # Save temporarily
            temp_file = os.path.join(SAVE_DIR, "temp_command.wav")
            write(temp_file, SAMPLE_RATE, trimmed_audio)

            # Transcribe
            self.listening_status.emit("Processing command...")
            text = self.transcription_engine.transcribe(temp_file).lower()

            # Find matching command
            matched_command = None
            for phrase, command in COMMAND_VOCABULARY.items():
                if phrase in text:
                    matched_command = command
                    break

            if matched_command:
                self.listening_status.emit(f"Command recognized: {matched_command}")
                self.command_detected.emit(matched_command)
            else:
                self.listening_status.emit(f"Unknown command: '{text}'")

        except Exception as e:
            self.listening_status.emit(f"Command processing error: {str(e)}")


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
            result = self.model.transcribe(audio_file, fp16=False)
            return result.get("text", "").strip()
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


import os
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QMessageBox
)


# Assume these utility classes are defined/imported from your project:
# AudioUtils, TranscriptionProcessor, and AudioRecorder.

class VoiceControlPrototypeApp(QWidget):
    """Main application window for the voice-based control prototype."""

    def __init__(self):
        super().__init__()
        AudioUtils.ensure_directories()

        self.file_counter = 1
        self.recording = False
        self.recorder_thread = None

        self._setup_ui()
        self._setup_transcription_engine()

    def _setup_ui(self):
        """Set up the user interface for the voice control prototype."""
        self.setWindowTitle("Voice Control Prototype")

        # Set fixed size with Android portrait (9:16) aspect ratio.
        base_width = 360
        self.setFixedSize(base_width, int(base_width * 16 / 9))

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 40, 20, 20)
        main_layout.setSpacing(20)

        # Title label
        title_label = QLabel("Voice Control Prototype")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333333;")
        main_layout.addWidget(title_label)

        # Toggle button for listening (recording) using the AudioRecorder.
        self.toggle_button = QPushButton("Voice\nMode")
        self.toggle_button.setFixedSize(80, 80)
        self.toggle_button.setStyleSheet("""
            QPushButton { 
                border-radius: 40px; 
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_recording)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.toggle_button)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # Status area
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #333333;")
        main_layout.addWidget(self.status_label)

        # Transcription result label
        self.transcription_label = QLabel("Transcription will appear here")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setWordWrap(True)
        self.transcription_label.setStyleSheet("font-size: 12px; color: #666666; padding: 10px;")
        main_layout.addWidget(self.transcription_label)

        self.setLayout(main_layout)

    def _setup_transcription_engine(self):
        """Set up the transcription engine for processing recordings."""
        self.transcription_processor = TranscriptionProcessor()
        self.transcription_processor.transcription_ready.connect(self.on_transcription_ready)
        self.transcription_processor.transcription_error.connect(self.on_transcription_error)
        self.transcription_processor.start()

    def toggle_recording(self):
        """Toggle the recording state using the AudioRecorder."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Begin recording speech using AudioRecorder."""
        self.recording = True
        self.update_toggle_button_state()
        self.status_label.setText("Listening for speech...")
        self.record_next()

    def stop_recording(self):
        """Stop recording speech."""
        self.recording = False
        self.update_toggle_button_state()
        self.status_label.setText("Listening stopped")
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop()

    def update_toggle_button_state(self):
        """Update the toggle button appearance based on the recording state."""
        if self.recording:
            self.toggle_button.setText("Exit\nvoice\nmode")
            self.toggle_button.setStyleSheet("""
                QPushButton { 
                    border-radius: 40px; 
                    background-color: #f44336;
                    color: white;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
        else:
            self.toggle_button.setText("Voice\nmode")
            self.toggle_button.setStyleSheet("""
                QPushButton { 
                    border-radius: 40px; 
                    background-color: #4CAF50;
                    color: white;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)

    def record_next(self):
        """Initiate the next audio recording session using AudioRecorder."""
        if not self.recording:
            return

        self.recorder_thread = AudioRecorder(self.file_counter)
        self.recorder_thread.finished.connect(self.on_recording_finished)
        self.recorder_thread.start()

    def on_recording_finished(self, result):
        """Callback when an audio recording is finished."""
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Error", str(result))
            self.stop_recording()
        elif result == "No speech detected":
            self.status_label.setText("No speech detected, listening again...")
            if self.recording:
                self.record_next()
        else:
            # Recording successful; process the file with the transcription engine.
            self.status_label.setText(f"Processing: {os.path.basename(result)}")
            self.file_counter += 1
            self.transcription_processor.add_file(result)

            if self.recording:
                self.record_next()

    def on_transcription_ready(self, file_path, text):
        """Display the transcription result for the recorded audio."""
        file_name = os.path.basename(file_path)
        self.status_label.setText(f"Transcribed: {file_name}")
        self.transcription_label.setText(text)
        print(f"Transcribed [{file_name}]: {text}")

    def on_transcription_error(self, file_path, error_msg):
        """Handle errors from the transcription engine."""
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
    window = VoiceControlPrototypeApp()
    window.show()
    sys.exit(app.exec_())
