# Grab DAX Voice Assistant – Handsfree AI Co-Pilot

Prototype presented by team Pikachu

## 🧠 Overview

This project is a **voice-controlled assistant prototype** designed for **Grab driver-partners (DAX)**, enabling **hands-free interactions** with the Grab platform. It empowers drivers with AI support in **noisy, real-world environments**, ensuring both **safety** and **productivity** on the road.

Built for the **UMHackathon 2025**, this solution addresses the *"Economic Empowerment through AI"* theme by allowing DAX users to:
- Navigate efficiently 🚗
- Accept ride requests 🛎️
- Chat with passengers 💬
- Mark passengers as fetched ✅
- Control these features via **voice commands** 🗣️

## 🎯 Problem Statement

Drivers currently rely on manual input or screen-based interfaces, which are unsafe while driving. This assistant solves that by:
- Supporting **voice-first interactions**
- Functioning in **challenging audio conditions**
- Adapting to **regional dialects, accents, and colloquialisms**
- Delivering **real-time transcription and intent detection**
- Providing **audio feedback** in local languages

## ✨ Features

- 🔊 **Noise-Resilient Voice Recorder** with real-time VAD + noise suppression
- 🧠 **Intent Prediction Engine** using language-detect + LLM (Gemma via Ollama)
- 📣 **Multilingual Text-to-Speech (TTS)** with support for English, Malay, and Chinese
- 🎨 Dual-theme GUI (Light & Dark modes)
- 📱 Android-style GUI with stacked pages and custom buttons
- 🔄 Context-aware navigation (back/home/intents)

## 🧩 Architecture
```
Voice Input → [Noise Reduction + VAD] → Whisper Transcription → → Language Detection → LLM Intent Classification → → UI Navigation / Voice Feedback (Edge-TTS)
```


## 🏗️ Modules

| Module | Description |
|--------|-------------|
| `audio_recorder.py` | Handles voice activity detection, noise reduction, and WAV recording |
| `transcription.py` | Uses OpenAI Whisper for speech-to-text |
| `intent_predictor.py` | LLM-based intent classification with multilingual prompt support |
| `tts_engine.py` | Uses Edge TTS for responsive speech synthesis |
| `main_app.py` | PyQt5 GUI with interactive pages and theme switching |

## 🧪 Key Technologies

- `PyQt5` – for GUI components
- `webrtcvad`, `noisereduce`, `sounddevice` – for audio preprocessing
- `whisper` – for transcription
- `langdetect`, `langchain`, `Ollama` – for language & intent modeling
- `pygame`, `edge-tts` – for multilingual TTS

## 🧠 AI Intelligence

- **Zero-Shot Intent Recognition** using `distilbert` (fallback)
- **Multilingual Prompt Templates** for language-specific intent grounding
- **Colloquial Slang & Accent Adaptability** (via LLM prompt tuning)

## 🌍 Environmental Resilience

Tested under:
- 🚦 Urban noise & engine rumble
- 🌧️ Rain & wind simulations
- 🔊 High-traffic scenarios (80–90 dB)
- 🇲🇾 Regional accents & speech quirks

## 🎥 Demo

A short demo video is included to illustrate:
- Voice interaction workflow
- Intent recognition
- Multilingual feedback
- UI transitions & safety logic

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Ollama

### Setup

download gemma3
```commandline
Ollama run gemma3:latest
```
Install Python Dependencies
```commandline
pip install -r srcs/requirements.txt
```

### Run the App

```bash
python srcs/main_app.py
```

## 🛡️ Safety & Ethics
- Hands-free only: No visual distractions for drivers
- Polite fallback prompts to clarify misheard commands
- Avoids unsafe instructions by design

## 📂 Directory Structure
```
srcs
├── main_app.py               # GUI logic
├── audio_recorder.py         # Audio capture and VAD
├── transcription.py          # Whisper-based STT
├── intent_predictor.py       # LLM-based intent detection
├── tts_engine.py             # Edge-TTS for feedback
└── data/                     # Recorded audio + TTS outputs
```

## 🧠 Future Improvements

- Next-Level Noise Reduction with Krisp API

- Switching to Locally trained Malaysian-Centric STT Model

- Adding AI Memory for Context awareness and personalization

## 📸 Snapshots

### 🟢 Normal Mode – Home Dashboard
Voice mode is off, and the driver can interact with the interface using buttons.

![normal mode](snapshots%2Fimg.png)

### 🎤 Voice Mode Activated
The app switches to dark theme when voice mode is active to signal listening state.

![voice mode](snapshots%2Fimg_1.png)

### 🧪 Example Voice Interactions

🧠 The assistant supports multilingual commands (English, Chinese, Malay) and can understand colloquial instructions, returning relevant actions with natural voice feedback.

```commandline
Transcribed [recorded_audio_1.wav]: Can you navigate me to the closest hospital?
[Edge-TTS] Speaking in en: Okay, navigating you to the closest hospital. Just one moment…
(Slight pause – simulating map lookup)
Okay, the closest hospital is Pusat Perubatan Universiti Malaya, approximately 5 kilometers away. I’m sending the route to your navigation system now.
Predicted intent: navigation
```
```
Transcribed [recorded_audio_1.wav]: 跟乘客讲模要到了就要到快点出来等
[Edge-TTS] Speaking in zh-cn: 好的，明白。
“好的，我来帮您跟乘客沟通。您说‘模要到了就要到快点出来等’，我来代替您说：‘乘客，请您尽快出来等待。’ 稍后我会提醒您注意安全。”
Predicted intent: chat_passenger
```
```
Transcribed [recorded_audio_2.wav]: 翻回主界面
[Edge-TTS] Speaking in cn: 好的，没问题。
“好的，正在返回主界面。”
Predicted intent: back
```
