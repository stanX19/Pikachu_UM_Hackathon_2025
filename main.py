import sys
import os
import queue
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import webrtcvad
import collections
import soundfile as sf

# Constants
SAVE_DIR = './data'
os.makedirs(SAVE_DIR, exist_ok=True)
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_TIMEOUT = 1.0  # seconds

class VADRecorder(QThread):
    finished = pyqtSignal(str)

    def __init__(self, file_index):
        super().__init__()
        self.vad = webrtcvad.Vad(2)
        self.buffer = collections.deque()
        self.silence_counter = 0
        self.running = True
        self.q = queue.Queue()
        self.file_index = file_index

    def stop(self):
        self.running = False

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def run(self):
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE,
                                   dtype='int16', channels=1, callback=self.callback):
                silent_chunks = 0
                voiced_chunks = 0
                collected_frames = []
                max_silence_chunks = int(SILENCE_TIMEOUT * 1000 / FRAME_DURATION)

                speech_started = False
                speech_start_index = 0

                while self.running:
                    frame = self.q.get()
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

                    if voiced_chunks > 5 and silent_chunks > max_silence_chunks:
                        break

            if voiced_chunks > 0:
                trimmed_frames = collected_frames[speech_start_index:]
                audio_data = b''.join(trimmed_frames)
                audio_np = np.frombuffer(audio_data, dtype='int16')
                file_path = os.path.join(SAVE_DIR, f'recorded_audio_{self.file_index}.wav')
                write(file_path, SAMPLE_RATE, audio_np)
                self.finished.emit(file_path)
            else:
                self.finished.emit("No speech detected")

        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class VoiceProcessor(QThread):
    def __init__(self, callback):
        super().__init__()
        self.queue = queue.Queue()
        self.callback = callback
        self.running = True

    def run(self):
        while self.running:
            try:
                file_path = self.queue.get(timeout=0.5)
                # insert model here
                text = file_path
                self.callback(text)
            except queue.Empty:
                continue

    def process_audio(self, file_path):
        try:
            audio, sr = sf.read(file_path)
            if sr != 16000:
                raise ValueError("Only 16kHz sample rate is supported")
            result = self.model(audio)
            return f"Transcribed [{file_path}]: {result}"
        except Exception as e:
            return f"Failed to process {file_path}: {e}"

    def add_file(self, path):
        self.queue.put(path)

    def stop(self):
        self.running = False

class AudioRecorderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice-Activated Recorder")
        self.setGeometry(100, 100, 300, 100)

        self.layout = QVBoxLayout()
        self.record_button = QPushButton("Voice Mode")
        self.record_button.setStyleSheet("QPushButton { border-radius: 40px; min-width: 80px; min-height: 80px; font-size: 16px; }")
        self.record_button.clicked.connect(self.toggle_recording)

        self.layout.addWidget(self.record_button)
        self.setLayout(self.layout)

        self.recording = False
        self.file_counter = 1
        self.thread = None

        self.voice_processor = VoiceProcessor(self.on_file_processed)
        self.voice_processor.start()

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_button.setText("Stop")
            self.record_next()
        else:
            self.recording = False
            self.record_button.setText("Voice Mode")
            if self.thread and self.thread.isRunning():
                self.thread.stop()

    def record_next(self):
        if not self.recording:
            self.record_button.setText("Voice Mode")
            self.record_button.setEnabled(True)
            return

        self.thread = VADRecorder(self.file_counter)
        self.thread.finished.connect(self.on_recording_finished)
        self.thread.start()

    def on_recording_finished(self, result):
        if result.startswith("Error"):
            QMessageBox.critical(self, "Error", result)
            self.recording = False
            self.record_button.setText("Voice Mode")
        elif result == "No speech detected":
            print("No speech detected, skipping file save.")
            if self.recording:
                self.record_next()
            else:
                self.record_button.setText("Voice Mode")
        else:
            print(f"Saved: {result}")
            self.file_counter += 1
            self.voice_processor.add_file(result)
            if self.recording:
                self.record_next()
            else:
                self.record_button.setText("Voice Mode")

    def on_file_processed(self, text):
        print(text)

    def closeEvent(self, event):
        self.voice_processor.stop()
        self.voice_processor.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioRecorderApp()
    window.show()
    sys.exit(app.exec_())
