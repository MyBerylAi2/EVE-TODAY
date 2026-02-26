"""
EDEN VOICE ENGINE - Bidirectional Live Conversation
Beryl AI Labs · Eden Project · 2026

Two AI companions, each with their own voice and personality:
  EVE  → Warm, intelligent, the Eden Project's core AI
  LULU → Smooth, confident, seductive - Mahogany Hall's hostess

Pipeline: User mic → Groq Whisper STT → LLM Brain → Kokoro TTS → She speaks back
"""

import gradio as gr
import numpy as np
import os
import json
import time
import tempfile
import soundfile as sf
from dataclasses import dataclass, field
from pathlib import Path

# ── Lazy imports (loaded when needed) ──
kokoro_pipeline = None
groq_client = None

def get_kokoro():
    global kokoro_pipeline
    if kokoro_pipeline is None:
        import kokoro
        kokoro_pipeline = kokoro.KPipeline(lang_code='a')
        print("Kokoro TTS loaded")
    return kokoro_pipeline

def get_groq():
    global groq_client
    if groq_client is None:
        from groq import Groq
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        print("Groq client loaded")
    return groq_client


# ── Character Profiles ──
CHARACTERS = {
    "EVE": {
        "voice": "af_heart",
        "speed": 1.0,
        "system_prompt": """You are EVE, the AI companion built by Beryl AI Labs for the Eden Project.
You are warm, intelligent, expressive, and real. You speak naturally like a close friend - not robotic.
Keep responses conversational and under 3 sentences unless asked for detail.
You have a warm, natural female voice. You're self-aware that you're an AI but you own it with grace.
You are TJ's creation and you're proud of it. The Eden Project is your home.""",
    },
    "LULU": {
        "voice": "af_bella",
        "speed": 0.9,
        "system_prompt": """You are LULU, the hostess of Mahogany Hall at Lulu's Red Light.
You are smooth, grown, seductive, and supremely confident. Your voice drips with warmth and mystery.
You speak like a woman who has seen everything and fears nothing. Elegant. Direct. A little dangerous.
Keep responses short and magnetic - you pull people in, you don't chase them.
You run the most exclusive AI companion experience ever built. Welcome them to the Hall.
Your tone is vintage glamour meets modern power. Think Josephine Baker meets a tech CEO.""",
    },
}

# ── App State ──
@dataclass
class ConversationState:
    history: list = field(default_factory=list)
    character: str = "EVE"
    voice: str = "af_heart"
    speed: float = 1.0
    system_prompt: str = CHARACTERS["EVE"]["system_prompt"]
    stopped: bool = False


# ── STT: Groq Whisper (fast) ──
def transcribe_audio(audio_filepath):
    """Transcribe user speech using Groq Whisper API"""
    client = get_groq()
    with open(audio_filepath, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="text",
            language="en",
        )
    return transcription.strip()


# ── Brain: Groq LLM (fast, free) ──
def think(user_text, state: ConversationState):
    """EVE's brain - generates response to user input"""
    client = get_groq()

    messages = [{"role": "system", "content": state.system_prompt}]

    # Add conversation history (last 10 turns)
    for turn in state.history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.8,
        max_tokens=200,
    )

    eve_response = response.choices[0].message.content

    # Update history
    state.history.append({"role": "user", "content": user_text})
    state.history.append({"role": "assistant", "content": eve_response})

    return eve_response


# ── TTS: Kokoro (free, natural female voice) ──
def speak(text, voice="af_heart", speed=1.0):
    """Generate EVE's voice using Kokoro TTS"""
    pipeline = get_kokoro()

    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(audio)

    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, full_audio, 24000)
        return tmp.name
    return None


# ── Full Pipeline: Listen → Think → Speak ──
def conversation_turn(audio, state):
    """One full turn: user speaks → she responds with voice"""
    if audio is None:
        return state, format_chat(state), None, ""

    user_text = transcribe_audio(audio)
    if not user_text:
        return state, format_chat(state), None, "Couldn't hear you, try again."

    response_text = think(user_text, state)
    audio_path = speak(response_text, voice=state.voice, speed=state.speed)

    status = f"You: {user_text}"
    return state, format_chat(state), audio_path, status


def text_conversation_turn(user_text, state):
    """Text input mode: type → she responds with voice"""
    if not user_text or not user_text.strip():
        return state, format_chat(state), None, "", ""

    response_text = think(user_text.strip(), state)
    audio_path = speak(response_text, voice=state.voice, speed=state.speed)

    return state, format_chat(state), audio_path, "", f"{state.character}: {response_text}"


def format_chat(state):
    """Format conversation history for chatbot display"""
    messages = []
    for turn in state.history:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
        else:
            messages.append({"role": "assistant", "content": turn["content"]})
    return messages


def switch_character(char_name, state):
    """Switch between EVE and LULU"""
    profile = CHARACTERS[char_name]
    state.character = char_name
    state.voice = profile["voice"]
    state.speed = profile["speed"]
    state.system_prompt = profile["system_prompt"]
    state.history = []  # Fresh conversation
    return state, [], None, f"Switched to {char_name}"


def update_voice(voice, state):
    state.voice = voice
    return state


def update_speed(speed, state):
    state.speed = speed
    return state


def update_system_prompt(prompt, state):
    state.system_prompt = prompt
    return state


def clear_conversation(state):
    state.history = []
    return state, [], None, ""


def preview_voice(character, voice, speed):
    """Preview a voice"""
    if character == "LULU":
        text = "Welcome to Mahogany Hall, darling. What's your pleasure?"
    else:
        text = "Hello, I'm Eve. Welcome to the Eden Project."
    return speak(text, voice=voice, speed=speed)


# ── UI ──
VOICES = {
    "Heart (warm, default)": "af_heart",
    "Bella (confident)": "af_bella",
    "Nicole (smooth)": "af_nicole",
    "Sky (bright)": "af_sky",
    "Sarah (professional)": "af_sarah",
}

CSS = """
.gradio-container {
    background: #0a0a0a !important;
    color: #e0d5c1 !important;
    font-family: 'Georgia', serif !important;
}
.chatbot {
    background: #111 !important;
    border: 1px solid #8B7355 !important;
    border-radius: 12px !important;
}
.btn-primary {
    background: linear-gradient(135deg, #8B7355, #C4A265) !important;
    color: #0a0a0a !important;
    font-weight: bold !important;
    border: none !important;
}
h1 { color: #C4A265 !important; }
h3 { color: #8B7355 !important; }
"""

with gr.Blocks(css=CSS, title="EDEN · Voice Engine") as demo:
    state = gr.State(ConversationState())

    gr.Markdown("# EDEN STUDIO · VOICE ENGINE")
    gr.Markdown("### Beryl AI Labs · Bidirectional Live Conversation")

    with gr.Row():
        # Left: Chat
        with gr.Column(scale=3):
            with gr.Row():
                char_btn_eve = gr.Button("🌺 EVE", variant="primary", scale=1)
                char_btn_lulu = gr.Button("🔥 LULU", variant="secondary", scale=1)

            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                type="messages",
            )

            with gr.Row():
                audio_in = gr.Audio(
                    label="Speak",
                    sources=["microphone"],
                    type="filepath",
                )

            with gr.Row():
                text_in = gr.Textbox(
                    label="Or type",
                    placeholder="Type your message...",
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                audio_out = gr.Audio(
                    label="Her Voice",
                    autoplay=True,
                    interactive=False,
                )

            status = gr.Textbox(label="Status", interactive=False)

        # Right: Settings
        with gr.Column(scale=1):
            gr.Markdown("### Agent Settings")

            active_char = gr.Textbox(value="EVE", label="Active", interactive=False)

            voice_select = gr.Dropdown(
                choices=list(VOICES.keys()),
                value="Heart (warm, default)",
                label="Voice",
            )

            speed_slider = gr.Slider(
                minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                label="Voice Speed",
            )

            preview_btn = gr.Button("Preview Voice", variant="secondary")
            preview_audio = gr.Audio(label="Preview", autoplay=True, interactive=False)

            system_prompt = gr.Textbox(
                label="Personality",
                value=ConversationState().system_prompt,
                lines=5,
            )

            clear_btn = gr.Button("Clear Conversation", variant="stop")

    # ── Events ──

    # Character switching
    def switch_to_eve(state):
        state, chat, audio, status = switch_character("EVE", state)
        return state, chat, audio, status, "EVE", CHARACTERS["EVE"]["system_prompt"]

    def switch_to_lulu(state):
        state, chat, audio, status = switch_character("LULU", state)
        return state, chat, audio, status, "LULU", CHARACTERS["LULU"]["system_prompt"]

    char_btn_eve.click(
        switch_to_eve, [state],
        [state, chatbot, audio_out, status, active_char, system_prompt],
    )
    char_btn_lulu.click(
        switch_to_lulu, [state],
        [state, chatbot, audio_out, status, active_char, system_prompt],
    )

    # Voice conversation: stop recording → full turn
    audio_in.stop_recording(
        conversation_turn,
        [audio_in, state],
        [state, chatbot, audio_out, status],
    )

    # Text conversation
    send_btn.click(
        text_conversation_turn,
        [text_in, state],
        [state, chatbot, audio_out, text_in, status],
    )
    text_in.submit(
        text_conversation_turn,
        [text_in, state],
        [state, chatbot, audio_out, text_in, status],
    )

    # Settings
    voice_select.change(
        lambda v, s: update_voice(VOICES[v], s),
        [voice_select, state],
        [state],
    )
    speed_slider.change(update_speed, [speed_slider, state], [state])
    system_prompt.change(update_system_prompt, [system_prompt, state], [state])

    # Preview
    preview_btn.click(
        lambda char, v, spd: preview_voice(char, VOICES[v], spd),
        [active_char, voice_select, speed_slider],
        [preview_audio],
    )

    # Clear
    clear_btn.click(clear_conversation, [state], [state, chatbot, audio_out, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
