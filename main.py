import sys
import os
import queue
import threading
import collections
import time
from typing import Optional, Callable, List, Tuple, Dict, Any

import numpy as np
import sounddevice as sd
import webrtcvad
import whisper
from scipy.io.wavfile import write
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox,
                             QLabel, QStackedLayout, QHBoxLayout, QSizePolicy, QFrame, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QIcon
import noisereduce

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


class LightColorPalette:
    """Light theme color palette for the application."""

    # Base colors
    background = "#FFFFFF"
    text_primary = "#333333"
    text_secondary = "#666666"

    # Button colors
    button_1 = "#2196F3"  # Nav button
    button_2 = "#4CAF50"  # Accept button
    button_3 = "#FF9800"  # Chat button
    button_4 = "#E91E63"  # Fetched button
    voice_button = "#673AB7"
    voice_button_hover = "#5E35B1"

    # Helper method to get button colors as a dict
    @classmethod
    def get_button_colors(cls) -> Dict[str, str]:
        return {
            "button_1": cls.button_1,
            "button_2": cls.button_2,
            "button_3": cls.button_3,
            "button_4": cls.button_4,
            "voice_button": cls.voice_button,
            "voice_button_hover": cls.voice_button_hover
        }

    # Helper method to get all theme colors as a dict
    @classmethod
    def get_theme(cls) -> Dict[str, str]:
        return {
            "app_bg": cls.background,
            "title_color": cls.text_primary,
            "nav_btn": cls.button_1,
            "accept_btn": cls.button_2,
            "chat_btn": cls.button_3,
            "fetched_btn": cls.button_4,
            "voice_btn_bg": cls.voice_button,
            "voice_btn_hover": cls.voice_button_hover,
            "status_color": cls.text_primary,
            "transcription_color": cls.text_secondary
        }


class DarkColorPalette:
    """Dark theme color palette for the application."""

    # Base colors
    background = "#1E1E2E"
    text_primary = "#E2E8F0"
    text_secondary = "#CBD5E1"

    # Button colors
    button_1 = "#3B82F6"  # Nav button
    button_2 = "#10B981"  # Accept button
    button_3 = "#F59E0B"  # Chat button
    button_4 = "#EC4899"  # Fetched button
    voice_button = "#EF4444"
    voice_button_hover = "#DC2626"

    # Helper method to get button colors as a dict
    @classmethod
    def get_button_colors(cls) -> Dict[str, str]:
        return {
            "button_1": cls.button_1,
            "button_2": cls.button_2,
            "button_3": cls.button_3,
            "button_4": cls.button_4,
            "voice_button": cls.voice_button,
            "voice_button_hover": cls.voice_button_hover
        }

    # Helper method to get all theme colors as a dict
    @classmethod
    def get_theme(cls) -> Dict[str, str]:
        return {
            "app_bg": cls.background,
            "title_color": cls.text_primary,
            "nav_btn": cls.button_1,
            "accept_btn": cls.button_2,
            "chat_btn": cls.button_3,
            "fetched_btn": cls.button_4,
            "voice_btn_bg": cls.voice_button,
            "voice_btn_hover": cls.voice_button_hover,
            "status_color": cls.text_primary,
            "transcription_color": cls.text_secondary
        }


class IntentPredictor:
    """Predicts intent from transcribed speech text."""

    # Define intents and their keywords/phrases
    INTENTS = {
        "navigation": ["navigate", "navigation", "directions", "map", "route", "go to", "take me", "bring me"],
        "accept_order": ["accept", "order", "pickup", "pick up", "new job", "new ride", "new customer"],
        "chat_passenger": ["chat", "message", "talk", "passenger", "customer", "client", "text", "say"],
        "fetched_passenger": ["fetched", "picked up", "got passenger", "passenger on board", "customer inside", "pick"],
        "exit_voice_mode": ["exit", "quit", "stop", "normal mode", "voice off"],
        "back": ["back", "menu", "leave"],
        "unknown": []  # Fallback intent
    }

    @classmethod
    def predict_intent(cls, text: str) -> str:
        """
        Predict the intent from the transcribed text.

        Args:
            text: The transcribed text from speech recognition

        Returns:
            The predicted intent as a string
        """
        text = text.lower()
        best_intent = "unknown"
        max_matches = 0

        # Check each intent for keyword matches
        for intent, keywords in cls.INTENTS.items():
            if intent == "unknown":
                continue

            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > max_matches:
                max_matches = matches
                best_intent = intent

        return best_intent


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

    @staticmethod
    def apply_noise_reduction(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction to the audio data."""
        try:

            # Use the first 0.5 seconds of audio as noise profile if available
            noise_sample_length = min(int(0.5 * sample_rate), len(audio_data) // 4)

            if noise_sample_length > 0:
                # Use the beginning portion of audio as noise profile
                noise_sample = audio_data[:noise_sample_length]
                reduced_audio = noisereduce.reduce_noise(
                    y=audio_data,
                    sr=sample_rate,
                    y_noise=noise_sample,
                    prop_decrease=1.00,
                    stationary=True
                )
            else:
                # If audio is too short, use statistical noise reduction
                reduced_audio = noisereduce.reduce_noise(
                    y=audio_data,
                    sr=sample_rate,
                    stationary=False
                )

            return reduced_audio
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return audio_data  # Return original if noise reduction fails


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
        self.buffer = collections.deque(maxlen=16)  # Buffer for noise reduction
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
        """Process incoming audio data with noise reduction and then VAD."""
        silent_chunks = 0
        voiced_chunks = 0
        max_voiced_chunks = 0
        collected_frames = []
        is_speech_frames = []  # Track which frames contain speech
        max_silence_chunks = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION)

        speech_started = False

        # Keep a sliding window of recent frames for noise profile
        noise_profile_frames = []

        while self.running:
            try:
                frame = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Add to sliding window of recent frames (for noise profile)
            noise_profile_frames.append(frame)
            if len(noise_profile_frames) > 8:  # Keep last ~240ms for noise profile
                noise_profile_frames.pop(0)

            # Convert current frame to numpy for noise reduction
            frame_np = np.frombuffer(frame, dtype='int16')

            # Only apply noise reduction if we have enough frames for a profile
            if len(noise_profile_frames) >= 4:
                try:
                    # Create noise profile from first few frames in the window
                    noise_profile = np.frombuffer(b''.join(noise_profile_frames[:3]), dtype='int16')

                    # Apply noise reduction to current frame
                    reduced_frame_np = noisereduce.reduce_noise(
                        y=frame_np,
                        sr=SAMPLE_RATE,
                        y_noise=noise_profile,
                        prop_decrease=1.00,
                        stationary=False
                    )

                    # Convert back to int16 if needed
                    if reduced_frame_np.dtype != np.int16:
                        reduced_frame_np = (reduced_frame_np * 32767).astype(np.int16)

                    # Convert back to bytes for VAD
                    reduced_frame = reduced_frame_np.tobytes()
                except Exception as e:
                    print(f"Noise reduction error on frame: {e}")
                    reduced_frame = frame  # Fall back to original frame
            else:
                reduced_frame = frame  # Not enough frames for noise profile yet

            # Apply VAD to the noise-reduced frame
            is_speech = self.vad.is_speech(reduced_frame, SAMPLE_RATE)
            is_speech_frames.append(is_speech)

            if is_speech and not speech_started:
                speech_started = True

            # Store the noise-reduced frame
            collected_frames.append(reduced_frame)

            if is_speech:
                silent_chunks = 0
                voiced_chunks += 1
                max_voiced_chunks = max(max_voiced_chunks, voiced_chunks)
            else:
                voiced_chunks = 0
                silent_chunks += 1

            # If we've detected speech and then sufficient silence, stop recording
            if max_voiced_chunks > 10 and silent_chunks > max_silence_chunks:
                break

        self._save_audio_if_speech_detected(collected_frames, is_speech_frames, max_voiced_chunks)

    def _save_audio_if_speech_detected(self, collected_frames, is_speech_frames, max_voiced_chunks):
        """Save the recorded audio if speech was detected, trimming silence from both ends."""
        if max_voiced_chunks <= 10:
            self.finished.emit("No speech detected")
            return

        try:
            # Convert noise-reduced frames to numpy array
            audio_data = b''.join(collected_frames)
            audio_np = np.frombuffer(audio_data, dtype='int16')

            # Trim silence from both ends
            trimmed_audio = AudioUtils.trim_silence(audio_np, is_speech_frames)

            # Additional check for minimum duration
            min_duration_samples = SAMPLE_RATE * 0.3  # at least 0.3 seconds
            if len(trimmed_audio) < min_duration_samples:
                print("Audio too short after trimming silence")
                self.finished.emit("Audio too short")
                return

            # Save to file
            file_path = AudioUtils.get_next_filename(self.file_index)
            write(file_path, SAMPLE_RATE, trimmed_audio)
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


class GridButton(QPushButton):
    """Custom button used in the app's grid layout."""

    def __init__(self, text, color, parent=None):
        super().__init__(text, parent)
        self.color = color
        self.setup_appearance()

    def setup_appearance(self):
        """Configure the button's visual appearance."""
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.update_style()

    def update_style(self, theme_color=None):
        """Update the button's style according to the current theme."""
        color = theme_color or self.color
        border_color = self.darken_color(color)
        hover_color = self.lighten_color(color)

        style = """
            QPushButton {
                background-color: %s;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 3px solid %s;
                border-radius: 15px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: %s;
            }
        """ % (color, border_color, hover_color)

        self.setStyleSheet(style)

    def darken_color(self, hex_color, amount=0.2):
        """Darken a hex color by a specified amount."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        r = max(0, int(r * (1 - amount)))
        g = max(0, int(g * (1 - amount)))
        b = max(0, int(b * (1 - amount)))

        return f"#{r:02x}{g:02x}{b:02x}"

    def lighten_color(self, hex_color, amount=0.2):
        """Lighten a hex color by a specified amount."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))

        return f"#{r:02x}{g:02x}{b:02x}"


class BackButton(QPushButton):
    """Custom back button for page navigation."""

    def __init__(self, parent=None):
        super().__init__("← Back", parent)
        self.setup_appearance()

    def setup_appearance(self):
        """Configure the button's visual appearance."""
        self.setFixedHeight(50)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def update_style(self, theme_color):
        """Update the button's style according to the given theme color."""
        border_color = self.darken_color(theme_color)
        hover_color = self.lighten_color(theme_color)

        style = f"""
            QPushButton {{
                background-color: {theme_color};
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 8px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """
        self.setStyleSheet(style)

    def darken_color(self, hex_color, amount=0.2):
        """Darken a hex color by a specified amount."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        r = max(0, int(r * (1 - amount)))
        g = max(0, int(g * (1 - amount)))
        b = max(0, int(b * (1 - amount)))

        return f"#{r:02x}{g:02x}{b:02x}"

    def lighten_color(self, hex_color, amount=0.2):
        """Lighten a hex color by a specified amount."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))

        return f"#{r:02x}{g:02x}{b:02x}"


class VoiceModeButton(QPushButton):
    """Custom round button for toggling voice mode."""

    def __init__(self, parent=None):
        super().__init__("Voice\nMode", parent)
        self.voice_mode_active = False
        self.setup_appearance()

    def setup_appearance(self):
        """Configure the button's visual appearance."""
        self.setFixedSize(80, 80)

    def update_style(self, theme):
        """Update the button's style according to the current theme."""
        bg_color = theme['voice_btn_bg']
        border_color = self.darken_color(bg_color)
        hover_color = theme['voice_btn_hover']

        style = f"""
            QPushButton {{ 
                border-radius: 40px; 
                background-color: {bg_color};
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 3px solid {border_color};
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """
        self.setStyleSheet(style)

    def toggle_state(self, is_active):
        """Toggle the button's state between normal and voice mode."""
        self.voice_mode_active = is_active
        text = "Exit\nVoice\nMode" if is_active else "Voice\nMode"
        self.setText(text)

    def darken_color(self, hex_color, amount=0.2):
        """Darken a hex color by a specified amount."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        r = max(0, int(r * (1 - amount)))
        g = max(0, int(g * (1 - amount)))
        b = max(0, int(b * (1 - amount)))

        return f"#{r:02x}{g:02x}{b:02x}"


class PageWidget(QWidget):
    """Widget representing a page in the application."""

    def __init__(self, title, color, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI for this page."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Back button at the top
        self.back_button = BackButton()
        self.back_button.update_style(self.color)
        layout.addWidget(self.back_button)

        # Title with page name
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: 24px; 
            font-weight: bold; 
            color: white;
            background-color: {self.color};
            border-radius: 10px;
            padding: 15px;
        """)
        layout.addWidget(title_label)

        # Content area - just a placeholder message
        content_label = QLabel(f"{self.title} Page Content\nto be developed...")
        content_label.setWordWrap(True)
        content_label.setAlignment(Qt.AlignCenter)
        content_label.setStyleSheet(f"""
            font-size: 18px;
            margin: 20px;
            padding: 30px;
            background-color: {self.lighten_color(self.color, 0.7)};
            border: 2px solid {self.color};
            border-radius: 15px;
        """)
        layout.addWidget(content_label)

        # Add stretcher to push everything up
        layout.addStretch()

        self.setLayout(layout)

    def lighten_color(self, hex_color, amount=0.2):
        """Lighten a hex color by a specified amount."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))

        return f"#{r:02x}{g:02x}{b:02x}"


class VoiceControlPrototypeApp(QWidget):
    """Main application window for the voice-based control prototype."""

    def __init__(self):
        super().__init__()
        AudioUtils.ensure_directories()

        self.file_counter = 1
        self.recording = False
        self.recorder_thread = None
        self.current_page = "home"

        # Define light and dark themes using our palette classes
        self.normal_theme = LightColorPalette.get_theme()
        self.voice_theme = DarkColorPalette.get_theme()
        self.current_theme = self.normal_theme

        self._setup_ui()
        self._setup_transcription_engine()
        self.apply_theme(self.normal_theme)

    def _setup_ui(self):
        """Set up the user interface for the voice control prototype."""
        self.setWindowTitle("Driver App")

        # Set fixed size with Android portrait (9:16) aspect ratio
        base_width = 360
        self.setFixedSize(base_width, int(base_width * 16 / 9))

        # Setup the stacked layout for multiple pages
        self.stacked_layout = QStackedLayout()

        # Create the home page
        self.home_page = QWidget()
        self._setup_home_page()
        self.stacked_layout.addWidget(self.home_page)

        # Initialize page dictionary to track our pages
        self.pages = {"home": 0}  # Maps page names to indices

        # Setup main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(self.stacked_layout)
        self.setLayout(main_layout)

    def _setup_home_page(self):
        """Set up the home page UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title label
        self.title_label = QLabel("Driver Assistant")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # 2x2 Grid of buttons
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        # Create the 2x2 grid buttons with our custom GridButton class
        self.nav_button = GridButton("Navigation", self.normal_theme["nav_btn"])
        self.accept_button = GridButton("Accept Order", self.normal_theme["accept_btn"])
        self.chat_button = GridButton("Chat with\nPassenger", self.normal_theme["chat_btn"])
        self.fetched_button = GridButton("Fetched\nPassenger", self.normal_theme["fetched_btn"])

        # Group our grid buttons for easy theming
        self.grid_buttons = {
            "navigation": self.nav_button,
            "accept_order": self.accept_button,
            "chat_passenger": self.chat_button,
            "fetched_passenger": self.fetched_button
        }

        # Connect button clicks
        self.nav_button.clicked.connect(
            lambda: self.navigate_to_page("Navigation", self.normal_theme["nav_btn"]))
        self.accept_button.clicked.connect(
            lambda: self.navigate_to_page("Accept Order", self.normal_theme["accept_btn"]))
        self.chat_button.clicked.connect(
            lambda: self.navigate_to_page("Chat with Passenger", self.normal_theme["chat_btn"]))
        self.fetched_button.clicked.connect(
            lambda: self.navigate_to_page("Fetched Passenger", self.normal_theme["fetched_btn"]))

        # Add buttons to the grid
        grid_layout.addWidget(self.nav_button, 0, 0)
        grid_layout.addWidget(self.accept_button, 0, 1)
        grid_layout.addWidget(self.chat_button, 1, 0)
        grid_layout.addWidget(self.fetched_button, 1, 1)

        layout.addLayout(grid_layout)

        # Voice mode button (below the grid)
        self.toggle_button = VoiceModeButton()
        self.toggle_button.clicked.connect(self.toggle_voice_mode)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.toggle_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Status area
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Transcription result label
        self.transcription_label = QLabel("Transcription will appear here")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setWordWrap(True)
        layout.addWidget(self.transcription_label)

        # Intent prediction label
        self.intent_label = QLabel("Intent: None")
        self.intent_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.intent_label)

        self.home_page.setLayout(layout)

    def navigate_to_page(self, title, color):
        """Navigate to a specific page or create it if it doesn't exist."""
        # Create a normalized page name for lookup
        page_name = title.lower().replace(" ", "_")

        # Check if the page already exists
        if page_name not in self.pages:
            # Create a new page
            new_page = PageWidget(title, color)
            new_page.back_button.clicked.connect(self.navigate_back)

            # Add to stacked layout and page dictionary
            index = self.stacked_layout.addWidget(new_page)
            self.pages[page_name] = index

        # Navigate to the page
        self.stacked_layout.setCurrentIndex(self.pages[page_name])
        self.current_page = page_name

        # Update status
        self.status_label.setText(f"Navigated to: {title}")
        print(f"Navigated to page: {title}")

    def navigate_back(self):
        """Navigate back to the home page."""
        self.stacked_layout.setCurrentIndex(self.pages["home"])
        self.current_page = "home"
        self.status_label.setText("Returned to home page")
        print("Navigated back to home page")

    def enter_voice_mode(self):
        """Enter voice mode - start listening and change theme."""
        if not self.recording:
            self.recording = True
            self.toggle_button.toggle_state(True)
            self.status_label.setText("Listening for speech...")
            self.apply_theme(self.voice_theme)
            self.record_next()

    def exit_voice_mode(self):
        """Exit voice mode - stop listening and restore normal theme."""
        if self.recording:
            self.recording = False
            self.toggle_button.toggle_state(False)
            self.status_label.setText("Voice mode exited")
            self.apply_theme(self.normal_theme)
            if self.recorder_thread and self.recorder_thread.isRunning():
                self.recorder_thread.stop()

    def toggle_voice_mode(self):
        """Toggle between voice mode and normal mode."""
        if not self.recording:
            self.enter_voice_mode()
        else:
            self.exit_voice_mode()

    def apply_theme(self, theme):
        """Apply the selected theme to all UI elements."""
        self.current_theme = theme

        # Apply background color to the app
        self.home_page.setStyleSheet(f"background-color: {theme['app_bg']};")

        # Apply colors to title
        self.title_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {theme['title_color']};")

        # Update grid buttons with new theme colors
        self.nav_button.update_style(theme["nav_btn"])
        self.accept_button.update_style(theme["accept_btn"])
        self.chat_button.update_style(theme["chat_btn"])
        self.fetched_button.update_style(theme["fetched_btn"])

        # Update voice mode button
        self.toggle_button.update_style(theme)

        # Apply colors to status and transcription labels
        self.status_label.setStyleSheet(f"font-size: 14px; color: {theme['status_color']};")
        self.transcription_label.setStyleSheet(
            f"font-size: 12px; color: {theme['transcription_color']}; padding: 10px;")
        self.intent_label.setStyleSheet(
            f"font-size: 12px; color: {theme['transcription_color']}; padding: 5px;")

    def _setup_transcription_engine(self):
        """Set up the transcription engine for processing recordings."""
        self.transcription_processor = TranscriptionProcessor()
        self.transcription_processor.transcription_ready.connect(self.on_transcription_ready)
        self.transcription_processor.transcription_error.connect(self.on_transcription_error)
        self.transcription_processor.start()

    def record_next(self):
        """Initiate the next audio recording session using AudioRecorder."""
        if not self.recording:
            return

        self.recorder_thread = AudioRecorder(self.file_counter)
        self.recorder_thread.finished.connect(self.on_recording_finished)
        self.recorder_thread.start()

    def handle_intent(self, intent, text):
        """Handle the detected intent from voice command."""
        # Map intents to button actions
        intent_to_action = {
            "navigation": lambda: self.navigate_to_page("Navigation", self.normal_theme["nav_btn"]),
            "accept_order": lambda: self.navigate_to_page("Accept Order", self.normal_theme["accept_btn"]),
            "chat_passenger": lambda: self.navigate_to_page("Chat with Passenger", self.normal_theme["chat_btn"]),
            "fetched_passenger": lambda: self.navigate_to_page("Fetched Passenger", self.normal_theme["fetched_btn"]),
            "exit_voice_mode": self.exit_voice_mode,
            "back": self.navigate_back
        }

        # Execute the action if the intent is recognized
        if self.current_page != "home" and intent != "unknown":
            self.navigate_back()

        if intent in intent_to_action:
            intent_to_action[intent]()
            return True

        return False

    def on_recording_finished(self, result):
        """Callback when an audio recording is finished."""
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Error", str(result))
            self.exit_voice_mode()
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
        """Display the transcription result and predict intent for the recorded audio."""
        file_name = os.path.basename(file_path)
        self.status_label.setText(f"Transcribed: {file_name}")
        self.transcription_label.setText(text)

        # Predict intent from the transcription
        intent = IntentPredictor.predict_intent(text)
        self.intent_label.setText(f"Intent: {intent}")

        # Handle the detected intent
        intent_handled = self.handle_intent(intent, text)

        # Print both transcription and intent prediction
        print(f"Transcribed [{file_name}]: {text}")
        print(f"Predicted intent: {intent}")

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