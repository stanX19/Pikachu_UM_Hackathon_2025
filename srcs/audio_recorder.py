import collections
import os
import queue
import sys
from typing import List

import noisereduce
import numpy as np
import sounddevice as sd
import webrtcvad
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.io.wavfile import write


SAVE_DIR = 'data'
SAMPLE_RATE = 32000
FRAME_DURATION = 30  # ms, can only be 10, 20 or 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
SILENCE_TIMEOUT = 1.0  # seconds
POST_SPEECH_BUFFER = 0.5  # seconds to keep after speech ends
VAD_AGGRESSIVENESS = 3  # 0-3, higher means more aggressive filtering
VOICE_CHUNKS_THRES = 5  # positive integer, minimum consecutive voiced chunk to be considered ppl speaking

# Voice command constants
COMMAND_TIMEOUT = 5  # seconds to listen for a command


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
                        prop_decrease=0.75,
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
            if max_voiced_chunks >= VOICE_CHUNKS_THRES and silent_chunks > max_silence_chunks:
                break

        self._save_audio_if_speech_detected(collected_frames, is_speech_frames, max_voiced_chunks)

    def _save_audio_if_speech_detected(self, collected_frames, is_speech_frames, max_voiced_chunks):
        """Save the recorded audio if speech was detected, trimming silence from both ends."""
        if max_voiced_chunks < VOICE_CHUNKS_THRES:
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
