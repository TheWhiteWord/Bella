import asyncio
from Models_interaction.faster_whisper_stt_tiny import capture_and_transcribe

async def capture_and_transcribe_audio():
    """Capture audio and return transcribed text with completion status."""
    text, is_complete = await capture_and_transcribe(debug=True)
    
    if text and 'watching' in text or "Let's go" in text:
        return None, False
    
    return text, is_complete
