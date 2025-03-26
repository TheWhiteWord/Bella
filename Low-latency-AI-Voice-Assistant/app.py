''''
DEPRECATED
This is a deprecated version of the app.py file. right now only using terminal
for interaction.
Kept for reference.
'''''

import streamlit as st
import asyncio
from utils.csm_tts import CSMSpeechProcessor
from Models_interaction.faster_whisper_stt_tiny import capture_audio, transcribe_audio
from utils.llm_interaction import generate_llm_response, get_available_models
import os
from pathlib import Path

st.set_page_config(
    page_title="Voice Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the CSM voice generator with Bella's voice
@st.cache_resource
def init_csm_tts():
    reference_audio = os.path.join("clone_files", "bella_edit.mp3")
    reference_text_path = os.path.join("clone_files", "Bella_transcript.txt")
    
    with open(reference_text_path, 'r') as f:
        reference_text = f.read()
    
    tts = CSMSpeechProcessor(
        reference_audio=reference_audio,
        reference_text=reference_text
    )
    return tts

# Initialize components
tts_engine = init_csm_tts()

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# UI elements for tunable parameters
st.sidebar.title("Model Settings")

# Get available models and check runtime availability
@st.cache_data(ttl=60)  # Cache for 60 seconds
async def get_model_status():
    return await get_available_models()

# Use asyncio to get model status
model_status = asyncio.run(get_model_status())

# Create a formatted display of models with availability
st.sidebar.subheader("Available Models")
for model_id, info in model_status.items():
    status_icon = "✅" if info['available'] else "❌"
    st.sidebar.markdown(f"""
    **{model_id}** {status_icon}
    - {info['description']}
    """)

# LLM Model selection with descriptions and availability
available_models = [m for m, info in model_status.items() if info['available']]
if not available_models:
    st.error("No Ollama models available. Please make sure Ollama is running with models installed.")
    st.stop()

llm_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    help="Choose the local Ollama model for response generation",
    format_func=lambda x: f"{x}"
)

# TTS settings
st.sidebar.markdown("---")
st.sidebar.subheader("Voice Settings")
voice_temperature = st.sidebar.slider(
    "Voice Temperature",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Higher values make the voice more expressive but less stable"
)

voice_speed = st.sidebar.slider(
    "Voice Speed",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Adjust the speaking speed of the voice"
)

# Main UI
st.title("Voice Assistant")
st.markdown(f"Using **{llm_model}** model with CSM voice synthesis")

# Status indicators
status_area = st.empty()
transcription_area = st.empty()
response_area = st.empty()
audio_area = st.empty()

# Add audio input
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

# Record button with visual feedback
record_col, status_col = st.columns([1, 4])
with record_col:
    if st.button("🎤 Record", use_container_width=True):
        with status_col:
            with st.spinner("Recording..."):
                audio_file = asyncio.run(capture_audio())
                if audio_file:
                    st.session_state.audio_file = audio_file
                    st.success("Recording complete!")
                else:
                    st.error("No speech detected")

# Process audio if available
if st.session_state.audio_file:
    with st.spinner("Processing..."):
        # Transcribe audio
        transcribed_text = asyncio.run(transcribe_audio(st.session_state.audio_file))
        if transcribed_text:
            transcription_area.write(f"🎤 You: {transcribed_text}")
            
            # Get conversation history as context
            history_context = "\n".join([
                f"User: {ex['user']}\nAssistant: {ex['assistant']}"
                for ex in st.session_state.conversation_history[-3:]  # Last 3 exchanges
            ])
            
            # Generate LLM response using Ollama
            with status_area:
                st.info(f"💭 Thinking...")
            
            response = asyncio.run(generate_llm_response(
                user_input=transcribed_text,
                history_context=history_context,
                model=llm_model
            ))
            
            if response:
                response_area.write(f"🤖 Assistant: {response}")
                
                # Generate speech using CSM
                with status_area:
                    st.info("🔊 Generating voice...")
                
                output_path = os.path.join("Testing", "audio files", "response.wav")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                audio, sample_rate = asyncio.run(tts_engine.generate_speech(response, output_path))
                
                if audio is not None:
                    audio_area.audio(output_path, sample_rate=sample_rate)
                    
                    # Update conversation history
                    st.session_state.conversation_history.append({
                        "user": transcribed_text,
                        "assistant": response,
                        "model": llm_model
                    })
            else:
                st.error("Failed to generate response. Make sure Ollama is running.")
            
            # Clear the audio file for next recording
            st.session_state.audio_file = None

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("### Conversation History")
    for i, exchange in enumerate(reversed(st.session_state.conversation_history), 1):
        st.markdown(f"""
        **Exchange {len(st.session_state.conversation_history) - i + 1}**
        - 🎤 You: {exchange['user']}
        - 🤖 Assistant ({exchange['model']}): {exchange['assistant']}
        ---
        """)

# Clear conversation button with confirmation
if st.session_state.conversation_history:
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        tts_engine.clear_context()  # Clear TTS context while keeping reference voice
        st.experimental_rerun()
