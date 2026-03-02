---
title: EVE - Talk to Me
emoji: 🔱
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: true
license: mit
---

# EVE — 4D Real-Time Conversational Avatar

**The Eden Project · Beryl AI Labs**

Talk to EVE. She listens, thinks, speaks, and shows her face.

## Pipeline
```
You speak → Whisper STT → Llama 3.3 Brain → Voice Engine → KDTalker Face → EVE responds
```

## Voice Engines
- **Qwen3-TTS** (default) — 97ms streaming, natural language voice design
- **Kokoro 82M** — <0.3s, af_heart warm female
- **Orpheus 3B** — ~200ms, emotion tags, human-like
- **Dia 1.6B** — Ultra-realistic dialogue with nonverbal
- **Chatterbox** — Voice cloning

## Features
- Voice input (microphone) or text input
- LLM-powered conversation (Llama 3.3 70B)
- 5 voice engines with auto-fallback
- Animated face responses (KDTalker)
- Conversation memory
- Warm, emotionally intelligent personality (HER 2013 inspired)

Built with the Eden Protocol by Beryl AI Labs.
