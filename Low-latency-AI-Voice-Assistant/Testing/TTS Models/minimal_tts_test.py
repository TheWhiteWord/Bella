import numpy as np
import sounddevice as sd
from RealtimeTTS import TextToAudioStream, KokoroEngine
import asyncio


# Now test TTS
async def main():
    engine = KokoroEngine(default_voice="af_heart")
    stream = TextToAudioStream(engine=engine)
    
    print("\nPlaying TTS...")
    stream.feed("Hello, this is a test.")
    stream.play_async()
    
    await asyncio.sleep(3)
    
    stream.stop()
    engine.shutdown()

asyncio.run(main())