---
title: EVE - Talk to Me
emoji: 🔱
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: true
license: mit
---

# EVE — 4D Real-Time Conversational Avatar

**The Eden Project · Beryl AI Labs**

Talk to EVE. She listens, thinks, speaks, and shows her face — in real time.

## What This Is

EVE is a fully functional AI conversational avatar deployed on HuggingFace Spaces. She has a brain (Llama 3.3 70B), a voice (5 TTS engines), and an animated face (KDTalker/MoDA). Two modes:

- **Live Mode** — WebRTC real-time voice conversation via FastRTC. Press Record, speak naturally, EVE responds with voice + animated face.
- **Chat Mode** — Type or use mic. Full pipeline with progressive streaming (hear EVE in ~5s, see animated face in ~30s).

## Pipeline
```
You speak → Whisper STT → Llama 3.3 70B Brain → Voice Engine → Face Animation → EVE responds
```

## Quick Start — Spin Up on HuggingFace

```bash
# 1. Clone this repo
git clone https://github.com/MyBerylAi2/EVE-TODAY.git
cd EVE-TODAY

# 2. Push to your HuggingFace Space
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/eve-voice-engine
git push hf main

# 3. Set your HF_TOKEN secret in Space settings
# Go to: Settings → Repository secrets → Add HF_TOKEN
```

Or if you already have the AIBRUH Space:

```bash
# Wake up the existing Space (unpause GPU + make public)
pip install huggingface_hub
python3 -c "
from huggingface_hub import HfApi; import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.update_repo_settings('AIBRUH/eve-voice-engine', private=False, repo_type='space')
api.restart_space('AIBRUH/eve-voice-engine')
print('EVE is waking up on T4 GPU...')
"
```

## Pause (Save GPU Credits)

```bash
python3 -c "
from huggingface_hub import HfApi; import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.pause_space('AIBRUH/eve-voice-engine')
api.update_repo_settings('AIBRUH/eve-voice-engine', private=True, repo_type='space')
print('EVE is sleeping. GPU off. Space private.')
"
```

## Voice Engines
| Engine | Latency | Style |
|--------|---------|-------|
| **Qwen3-TTS** (default) | 97ms streaming | Natural language voice design |
| **Kokoro 82M** | <0.3s | af_heart warm female (used in Live Mode) |
| **Orpheus 3B** | ~200ms | Human-like with emotion tags |
| **Dia 1.6B** | Batch | Ultra-realistic dialogue with (laughs) (sighs) |
| **Chatterbox** | Sub-200ms | Voice cloning |

## Face Animation (Cascade Fallback)
1. **KDTalker** — Full face expression, ~20s
2. **MoDA** — Emotion-controlled (Happiness), ~10s
3. **LTX2** — Prompt-driven video generation, ~15s

## Tech Stack
- Gradio 5.50 + FastRTC (WebRTC)
- HuggingFace Inference API (Llama 3.3 70B)
- External Gradio Spaces for TTS + face animation
- T4 GPU (16GB VRAM)

## Files
- `app.py` — Main application (~1600 lines)
- `requirements.txt` — Dependencies
- `eve-portrait.jpeg` — EVE's portrait image
- `voices/` — Reference voice samples for cloning

## Current Status
- Live Mode: WebRTC mic + face display + transcript + idle animation
- Chat Mode: Full pipeline confirmed working
- Progressive streaming: hear audio in ~5s, see animated face in ~30s

## What's Next
- Faster face animation (local model on T4 instead of external Spaces)
- True real-time video streaming in Live Mode
- More expressive idle animations

Built with the Eden Protocol by Beryl AI Labs.
