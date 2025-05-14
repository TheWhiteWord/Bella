#!/usr/bin/env python3
"""
Test script to play TTS audio from a text file using KokoroTTSWrapper.
This will help test the visualizer with real audio output.

Usage:
    python test_kokoro_tts_play.py input.txt

Where input.txt is a plain text file with the text to synthesize and play.
"""
import sys
import asyncio
from kokoro_tts import KokoroTTSWrapper

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_kokoro_tts_play.py input.txt")
        sys.exit(1)
    text_file = sys.argv[1]
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        print("Input file is empty.")
        sys.exit(1)
    tts = KokoroTTSWrapper()
    await tts.generate_speech(text)

if __name__ == "__main__":
    asyncio.run(main())
