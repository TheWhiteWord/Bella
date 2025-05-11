"""
Test for verifying the model loading path in faster_whisper_stt_tiny.py.
"""


import sys
import os
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))
from audio.whisper import faster_whisper_stt_tiny



def test_get_whisper_model_local():
    """Test that get_whisper_model loads the local model if available."""
    class DummyModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    called = {}
    def dummy_whisper_model(*args, **kwargs):
        called['args'] = args
        called['kwargs'] = kwargs
        return DummyModel(*args, **kwargs)

    with patch.object(faster_whisper_stt_tiny, 'WhisperModel', dummy_whisper_model):
        # Clear lru_cache before test
        faster_whisper_stt_tiny.get_whisper_model.cache_clear()
        model = faster_whisper_stt_tiny.get_whisper_model()
        assert called['args'][0] == faster_whisper_stt_tiny.MODEL_NAME, f"Expected model name {faster_whisper_stt_tiny.MODEL_NAME}, got {called['args'][0]}"
        assert called['kwargs']['device'] == 'gpu'
        assert isinstance(model, DummyModel)



if __name__ == "__main__":
    try:
        test_get_whisper_model_local()
        print("test_get_whisper_model_local passed.")
    except AssertionError as e:
        print(f"test_get_whisper_model_local failed: {e}")
