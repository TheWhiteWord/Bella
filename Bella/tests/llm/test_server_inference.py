import asyncio
import time
import os
import sys
from openai import AsyncOpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "http://localhost:8080/v1"
API_KEY = "none"
# Target the specific model causing issues
MODEL_NAME = "Llama-3.1-8B-Lexi-Uncensored_V2_Q8" 

async def test_server_inference():
    """Test connectivity and inference speed of the Llama.cpp server."""
    
    logger.info(f"Connecting to Llama.cpp server at {BASE_URL}...")
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    # 1. Test Connectivity & List Models
    try:
        start_time = time.time()
        logger.info("checking available models...")
        models_response = await client.models.list()
        elapsed = time.time() - start_time
        
        available_models = [m.id for m in models_response.data]
        logger.info(f"Successfully listed models in {elapsed:.2f}s")
        logger.info(f"Available models: {available_models}")
        
        if MODEL_NAME not in available_models:
            logger.error(f"❌ Target model '{MODEL_NAME}' not found in available models!")
            return
        else:
            logger.info(f"✅ Target model '{MODEL_NAME}' found.")

    except Exception as e:
        logger.error(f"❌ Failed to connect or list models: {e}")
        return

    # 2. Test Inference (Simple)
    logger.info(f"\nTesting inference with model: {MODEL_NAME}")
    prompt = "Hello! briefly introduce yourself."
    messages = [{"role": "user", "content": prompt}]
    
    try:
        start_time = time.time()
        logger.info("Sending request...")
        
        # We start with a generous timeout to catch slow responses vs actual timeouts
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=50 
            ),
            timeout=120.0
        )
        
        elapsed = time.time() - start_time
        content = response.choices[0].message.content
        
        logger.info(f"✅ Response received in {elapsed:.2f}s")
        logger.info(f"Output: {content}")
        
    except asyncio.TimeoutError:
        logger.error("❌ Request Timed Out (after 120s)")
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_server_inference())
    except KeyboardInterrupt:
        logger.info("Test stopped by user.")
