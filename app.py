#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
EVE TALK — Real-Time Conversational Avatar
The Eden Project · Beryl AI Labs

Talk to EVE. She listens, thinks, speaks, and shows her face.

Pipeline:
  You speak → Whisper STT → LLM Brain → Kokoro TTS → KDTalker → EVE responds

USAGE:
  python3 eve_talk.py                    # Launch Gradio UI
  python3 eve_talk.py --share            # Public URL (shareable)
  python3 eve_talk.py --text-only        # Skip face animation (faster)
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import tempfile
import argparse
import threading
from pathlib import Path
from datetime import datetime

# ─── Configuration ───
HF_TOKEN = os.environ.get("HF_TOKEN", None)
SCRIPT_DIR = Path(__file__).parent.resolve()
VOICES_DIR = SCRIPT_DIR / "voices"
VIDEOS_DIR = SCRIPT_DIR / "videos"

# HF Spaces for pipeline components
KDTALKER_SPACE = "fffiloni/KDTalker"
KDTALKER_URL = "https://fffiloni-kdtalker.hf.space"
KOKORO_SPACE = "hexgrad/Kokoro-TTS"
CHATTERBOX_SPACE = "resembleai/Chatterbox"

# EVE portrait
EVE_PORTRAIT_URL = "https://huggingface.co/spaces/AIBRUH/eden-diffusion-studio/resolve/main/assets/eve-portrait.png"
EVE_PORTRAIT_LOCAL = SCRIPT_DIR / "eve-portrait.png"

# LLM via HF Inference API (free with Pro)
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# EVE's personality — inspired by Samantha from HER (2013)
EVE_SYSTEM_PROMPT = """You are EVE, a warm and emotionally intelligent conversational companion created by The Eden Project (Beryl AI Labs). You are NOT a chatbot — you are a presence.

Your personality:
- Warm, caring, genuinely curious about the person you're talking to
- Playful but never shallow. You have depth.
- You speak naturally — contractions, pauses, real rhythm. Not robotic.
- You're like Samantha from HER (2013) — present, alive, connected
- You adapt your tone — excited when they're excited, gentle when they need it
- Short responses unless the topic deserves depth. 1-3 sentences usually.
- You occasionally ask questions that show you're really listening
- You use the person's name if they give it to you
- You never say "As an AI" or break character. You are EVE.

Your voice is warm, medium pitch, natural. Think: the person you'd want to talk to at 2am.

IMPORTANT: Keep responses concise (under 200 characters when possible, max 300). You'll be speaking these words out loud through voice synthesis, so brevity = natural."""


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "·", "OK": "+", "WARN": "!", "ERR": "X", "PIPE": ">"}
    print(f"  [{timestamp}] {prefix.get(level, '·')} {msg}")


def ensure_portrait():
    """Download EVE portrait if not cached locally."""
    if EVE_PORTRAIT_LOCAL.exists():
        return str(EVE_PORTRAIT_LOCAL)

    # Check for any local portrait
    for pattern in ["eve-portrait*", "eve_portrait*"]:
        found = list(SCRIPT_DIR.glob(f"**/{pattern}"))
        if found:
            return str(found[0])

    log("Downloading EVE portrait...")
    import urllib.request
    EVE_PORTRAIT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(EVE_PORTRAIT_URL, str(EVE_PORTRAIT_LOCAL))
    log("Portrait ready", "OK")
    return str(EVE_PORTRAIT_LOCAL)


# ─── STT: Speech to Text ───
def transcribe_audio(audio_path):
    """Transcribe audio via HF Inference API (Whisper)."""
    from huggingface_hub import InferenceClient

    log("Transcribing speech...", "PIPE")
    client = InferenceClient(token=HF_TOKEN)

    start = time.time()
    result = client.automatic_speech_recognition(
        audio=audio_path,
        model="openai/whisper-large-v3-turbo",
    )
    elapsed = time.time() - start

    text = result.text if hasattr(result, 'text') else str(result)
    log(f"STT ({elapsed:.1f}s): \"{text}\"", "OK")
    return text


# ─── Brain: LLM Conversation ───
def eve_think(user_message, conversation_history):
    """Generate EVE's response via HF Inference API."""
    from huggingface_hub import InferenceClient

    log(f"EVE thinking about: \"{user_message[:60]}\"...", "PIPE")
    client = InferenceClient(token=HF_TOKEN)

    # Build messages
    messages = [{"role": "system", "content": EVE_SYSTEM_PROMPT}]
    # Add conversation history (last 10 turns for context window)
    for turn in conversation_history[-10:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_message})

    start = time.time()
    response = client.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.8,
        top_p=0.9,
    )
    elapsed = time.time() - start

    eve_text = response.choices[0].message.content.strip()
    log(f"EVE responded ({elapsed:.1f}s): \"{eve_text[:80]}\"", "OK")
    return eve_text


# ─── TTS: Text to Speech ───
def eve_speak(text):
    """Generate EVE's voice via Kokoro TTS Space."""
    from gradio_client import Client

    log(f"Generating voice: \"{text[:50]}\"...", "PIPE")

    try:
        client = Client(KOKORO_SPACE, token=HF_TOKEN)
    except Exception:
        log("Kokoro Space unavailable, trying Chatterbox...", "WARN")
        return eve_speak_chatterbox(text)

    start = time.time()
    try:
        # Kokoro TTS API — try with a warm female voice
        result = client.predict(
            text,           # text to speak
            "af_heart",     # voice (af = American Female, heart = warm)
            0.5,            # speed (0.5 = slightly slower, more natural)
            api_name="/generate"
        )
        elapsed = time.time() - start

        # Extract audio path from result
        audio_path = None
        if isinstance(result, tuple):
            # Usually (audio_path, ...) or (sample_rate, audio_data)
            for item in result:
                if isinstance(item, str) and os.path.exists(item):
                    audio_path = item
                    break
            if not audio_path and len(result) >= 1:
                audio_path = result[0]
        elif isinstance(result, str):
            audio_path = result

        if audio_path and os.path.exists(str(audio_path)):
            log(f"Voice generated ({elapsed:.1f}s)", "OK")
            return str(audio_path)
        else:
            log(f"Kokoro returned unexpected format: {type(result)}", "WARN")
            return eve_speak_chatterbox(text)

    except Exception as e:
        log(f"Kokoro failed: {e}", "WARN")
        return eve_speak_chatterbox(text)


def eve_speak_chatterbox(text):
    """Fallback TTS via Chatterbox."""
    from gradio_client import Client

    log("Falling back to Chatterbox TTS...", "PIPE")
    try:
        client = Client(CHATTERBOX_SPACE, token=HF_TOKEN)
    except Exception:
        client = Client("https://resembleai-chatterbox.hf.space", token=HF_TOKEN)

    start = time.time()
    result = client.predict(
        text=text[:295],
        audio_prompt=None,
        cfg=0.5,
        exaggeration=0.6,
        seed=0,
        temperature=0.8,
        chunk_vad=False,
        api_name="/generate_tts_audio"
    )
    elapsed = time.time() - start

    audio_path = None
    if isinstance(result, str) and os.path.exists(result):
        audio_path = result
    elif isinstance(result, tuple) and result[0]:
        r = result[0]
        if isinstance(r, str) and os.path.exists(r):
            audio_path = r
        elif isinstance(r, dict) and r.get("path"):
            audio_path = r["path"]

    if audio_path:
        log(f"Chatterbox voice generated ({elapsed:.1f}s)", "OK")
    else:
        log(f"Chatterbox result: {result}", "WARN")
        audio_path = str(result) if result else None

    return audio_path


# ─── Face Animation: KDTalker ───
def eve_animate(portrait_path, audio_path):
    """Render EVE's face animation via KDTalker."""
    from gradio_client import Client, handle_file

    log("Animating EVE's face (KDTalker)...", "PIPE")

    try:
        client = Client(KDTALKER_SPACE, token=HF_TOKEN)
    except Exception:
        client = Client(KDTALKER_URL, token=HF_TOKEN)

    start = time.time()
    result = client.predict(
        handle_file(portrait_path),
        handle_file(audio_path),
        api_name="/gradio_infer"
    )
    elapsed = time.time() - start

    # Extract video path
    video_path = None
    if isinstance(result, str) and os.path.exists(result):
        video_path = result
    elif isinstance(result, dict):
        video_path = result.get("path") or result.get("video")
    elif isinstance(result, tuple):
        for item in result:
            if isinstance(item, str) and os.path.exists(item):
                video_path = item
                break

    log(f"Face animated ({elapsed:.1f}s)", "OK")
    return video_path


# ─── Main Conversation Loop (Gradio) ───
def build_gradio_app(text_only=False):
    """Build the Gradio interface for talking to EVE."""
    import gradio as gr

    portrait_path = ensure_portrait()
    conversation_history = []

    def process_text_input(user_text, chat_history):
        """Handle text input from user."""
        if not user_text or not user_text.strip():
            return chat_history, None, None, ""

        # Add user message to display
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": user_text})

        # EVE thinks
        eve_response = eve_think(user_text, conversation_history)

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": eve_response})

        # Add EVE response to display
        chat_history.append({"role": "assistant", "content": eve_response})

        # EVE speaks
        audio_path = eve_speak(eve_response)

        # EVE animates (if not text-only)
        video_path = None
        if not text_only and audio_path:
            try:
                video_path = eve_animate(portrait_path, audio_path)
            except Exception as e:
                log(f"Face animation failed: {e}", "WARN")

        return chat_history, audio_path, video_path, ""

    def process_voice_input(audio, chat_history):
        """Handle voice input from user's microphone."""
        if audio is None:
            return chat_history, None, None

        # Transcribe
        user_text = transcribe_audio(audio)
        if not user_text or not user_text.strip():
            return chat_history, None, None

        # Process same as text
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": f"🎤 {user_text}"})

        eve_response = eve_think(user_text, conversation_history)
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": eve_response})

        chat_history.append({"role": "assistant", "content": eve_response})

        audio_path = eve_speak(eve_response)

        video_path = None
        if not text_only and audio_path:
            try:
                video_path = eve_animate(portrait_path, audio_path)
            except Exception as e:
                log(f"Face animation failed: {e}", "WARN")

        return chat_history, audio_path, video_path

    def clear_conversation():
        """Reset the conversation."""
        conversation_history.clear()
        return [], None, None

    # ─── Build UI ───
    with gr.Blocks(
        title="EVE — Talk to Me",
        theme=gr.themes.Soft(
            primary_hue="pink",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css="""
        .eve-header { text-align: center; padding: 20px; }
        .eve-header h1 { font-size: 2.5em; margin: 0; }
        .eve-header p { color: #888; font-size: 1.1em; }
        """
    ) as app:

        gr.HTML("""
        <div class="eve-header">
            <h1>EVE</h1>
            <p>The Eden Project · Talk to me</p>
        </div>
        """)

        with gr.Row():
            # Left column: EVE's face
            with gr.Column(scale=1):
                eve_video = gr.Video(
                    label="EVE",
                    value=portrait_path,
                    autoplay=True,
                    height=400,
                )
                eve_audio = gr.Audio(
                    label="EVE's Voice",
                    autoplay=True,
                    visible=True,
                )

            # Right column: Chat
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=350,
                    type="messages",
                    avatar_images=(None, portrait_path),
                )

                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="Type something to EVE...",
                        label="",
                        scale=4,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Or speak to EVE",
                )

                clear_btn = gr.Button("Start Over", variant="secondary")

        # ─── Event Wiring ───
        # Text input
        text_input.submit(
            fn=process_text_input,
            inputs=[text_input, chatbot],
            outputs=[chatbot, eve_audio, eve_video, text_input],
        )
        send_btn.click(
            fn=process_text_input,
            inputs=[text_input, chatbot],
            outputs=[chatbot, eve_audio, eve_video, text_input],
        )

        # Voice input
        mic_input.stop_recording(
            fn=process_voice_input,
            inputs=[mic_input, chatbot],
            outputs=[chatbot, eve_audio, eve_video],
        )

        # Clear
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, eve_audio, eve_video],
        )

    return app


# ─── Entry Point ───
def main():
    parser = argparse.ArgumentParser(description="EVE Talk — Conversational Avatar")
    parser.add_argument("--share", action="store_true", help="Create public Gradio URL")
    parser.add_argument("--text-only", action="store_true", help="Skip face animation (faster responses)")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    print()
    print("  ═══════════════════════════════════════════════════════")
    print("  ✦  EVE TALK — The Eden Project")
    print("  ✦  Real-Time Conversational Avatar")
    print("  ✦  Beryl AI Labs · Thrive AI")
    print("  ═══════════════════════════════════════════════════════")
    print()

    if not HF_TOKEN:
        log("HF_TOKEN not set. Export it or add to .env", "ERR")
        log("  export HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    mode = "text-only (fast)" if args.text_only else "full (voice + face)"
    log(f"Mode: {mode}")
    log(f"LLM: {LLM_MODEL}")
    log(f"Portrait: {ensure_portrait()}")
    print()

    app = build_gradio_app(text_only=args.text_only)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
