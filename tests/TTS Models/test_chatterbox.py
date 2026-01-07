import os
import sys
import time
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatterbox_test")

def benchmark_chatterbox():
    # Paths
    project_root = "/media/theww/AI/Code/AI/Bella/Bella"
    reference_audio = os.path.join(project_root, "src/audio/clone_files/her_clone_short_15s_24k.wav")
    output_audio = os.path.join(project_root, "results/test_results/tts_tests/output_chatterbox.wav")
    
    os.makedirs(os.path.dirname(output_audio), exist_ok=True)

    logger.info("Initializing Chatterbox-Turbo TTS on CUDA...")
    start_init = time.time()
    try:
        model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    init_time = time.time() - start_init
    logger.info(f"Model loaded in {init_time:.2f}s")

    test_text = "Hello, I am testing the latency of this model."
    
    # Warm-up run
    logger.info("Performing warm-up run...")
    model.generate(test_text, audio_prompt_path=reference_audio)

    # Short sentence test
    test_text_short = "Yes, I agree."
    logger.info(f"Benchmarking 'warm' short sentence generation: {test_text_short}")
    start_gen = time.time()
    wav = model.generate(test_text_short, audio_prompt_path=reference_audio)
    gen_time = time.time() - start_gen
    logger.info(f"Short generation completed in {gen_time:.2f}s")
    
    duration = wav.shape[-1] / model.sr
    logger.info(f"Audio duration: {duration:.2f}s")
    logger.info(f"Real Time Factor (RTF): {gen_time / duration:.4f}")

    # Save the output
    ta.save(output_audio, wav.cpu(), model.sr)
    logger.info(f"Result saved to: {output_audio}")

if __name__ == "__main__":
    benchmark_chatterbox()
