# EVE Voice Engine — Claude Code Project Context

## Project
EVE is a real-time 4D conversational avatar on HuggingFace Spaces.
- **HF Space**: AIBRUH/eve-voice-engine (may be PAUSED/PRIVATE — see wake commands below)
- **GitHub mirror**: https://github.com/MyBerylAi2/EVE-TODAY
- **SDK**: Gradio 5.50.0 on T4 GPU

## Wake Up EVE
```python
from huggingface_hub import HfApi; import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.update_repo_settings('AIBRUH/eve-voice-engine', private=False, repo_type='space')
api.restart_space('AIBRUH/eve-voice-engine')
```

## Pause EVE (Save GPU)
```python
from huggingface_hub import HfApi; import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.pause_space('AIBRUH/eve-voice-engine')
api.update_repo_settings('AIBRUH/eve-voice-engine', private=True, repo_type='space')
```

## Architecture
- Brain: Llama 3.3 70B via HF Inference API
- Voice: 5 TTS engines (Kokoro for live, Qwen3 for chat)
- Face: KDTalker → MoDA → LTX2 cascade (external Gradio Spaces)
- Live Mode: FastRTC WebRTC with ReplyOnPause
- Chat Mode: Progressive streaming generators (two yields)

## Key Rules
- NEVER use SadTalker
- NEVER show placeholders — always real content or fallback
- Full face expression, not just lip sync
- Clean portrait UI — no CSS animations on face
- Mic stays on until user turns it off
- gradio_client uses `hf_token=` not `token=` (Gradio 5.29+)
- FastRTC needs `fastrtc[vad]` not just `fastrtc`
- Use `get_cloudflare_turn_credentials` not deprecated `get_hf_turn_credentials`

## Push Flow
```bash
git push origin main   # → HuggingFace Space (auto-rebuilds)
git push github main   # → GitHub mirror
```
