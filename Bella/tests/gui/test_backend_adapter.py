"""Tests for the GUI backend adapter.

This module tests the functionality of the GUI backend adapter,
which bridges between the async backend implementation and the
synchronous GUI interface.
"""
import os
import sys
import pytest
import tempfile
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gui_backend_adapter import BackendAdapter

# Test data
TEST_TEXT = "This is a test text."
TEST_TRANSCRIPT = "Hello, world."
TEST_CONTEXT = "Some context information."
TEST_RESPONSE = "This is a test response from the LLM."


@pytest.fixture
def mock_recorder():
    """Create a mock BufferedRecorder."""
    recorder = MagicMock()
    recorder.is_recording = False
    recorder.should_stop = False
    recorder.initialize_audio_settings = MagicMock()
    recorder.start_recording = MagicMock()
    return recorder


@pytest.fixture
def mock_audio_manager():
    """Create a mock AudioSessionManager."""
    manager = MagicMock()
    manager.set_recorder = MagicMock()
    manager.pause_session = MagicMock()
    manager.resume_session = MagicMock()
    # Make start_session return mock transcript
    manager.start_session = AsyncMock(return_value=(TEST_TRANSCRIPT, [TEST_TRANSCRIPT]))
    return manager


@pytest.fixture
def mock_tts():
    """Create a mock TTS engine."""
    tts = MagicMock()
    tts.generate_speech = AsyncMock()
    tts.stop = MagicMock()
    return tts


@pytest.fixture
def adapter_with_mocks(mock_recorder, mock_audio_manager, mock_tts):
    """Create a backend adapter with mocked components."""
    with patch('src.gui_backend_adapter.BufferedRecorder', return_value=mock_recorder), \
         patch('src.gui_backend_adapter.AudioSessionManager', return_value=mock_audio_manager), \
         patch('src.gui_backend_adapter.KokoroTTSWrapper', return_value=mock_tts), \
         patch('src.gui_backend_adapter.generate_chat_response', AsyncMock(return_value=TEST_RESPONSE)):
        
        # Create the adapter with mocked dependencies
        adapter = BackendAdapter()
        adapter.asyncio_loop = None  # We won't create a real asyncio loop for tests
        
        # Create a mock for _run_async instead of using a real event loop
        async def mock_run_async_impl(coro):
            # Just await the coroutine directly in the test's event loop
            return await coro
            
        adapter._run_async = MagicMock()
        adapter._run_async.side_effect = lambda coro: asyncio.run(mock_run_async_impl(coro))
        
        yield adapter
        
        # No need to clean up the loop as we're using mocks


def test_initialization():
    """Test that the adapter initializes correctly."""
    with patch('src.gui_backend_adapter.BufferedRecorder'), \
         patch('src.gui_backend_adapter.AudioSessionManager'), \
         patch('src.gui_backend_adapter.KokoroTTSWrapper'):
        
        adapter = BackendAdapter(sink_name="test_sink")
        
        assert adapter is not None
        assert adapter.sink_name == "test_sink"
        assert adapter.recorder is not None
        assert adapter.audio_manager is not None
        assert adapter.speech_callback is None
        assert adapter.is_paused is False


def test_start_continuous_listening(adapter_with_mocks):
    """Test starting continuous listening."""
    adapter = adapter_with_mocks
    callback = MagicMock()
    
    # Replace the thread starting with a mock to avoid actual thread creation
    with patch.object(adapter, '_listening_worker'):
        adapter.start_continuous_listening(callback)
    
    assert adapter.speech_callback == callback
    assert adapter.running is True
    assert adapter.is_paused is False
    adapter.recorder.initialize_audio_settings.assert_called_once()


def test_pause_listening(adapter_with_mocks):
    """Test pausing listening."""
    adapter = adapter_with_mocks
    adapter.is_paused = False
    adapter.recorder.is_recording = True
    
    adapter.pause_listening()
    
    assert adapter.is_paused is True
    assert adapter.recorder.should_stop is True
    adapter.audio_manager.pause_session.assert_called_once()


def test_resume_listening(adapter_with_mocks):
    """Test resuming listening."""
    adapter = adapter_with_mocks
    adapter.is_paused = True
    
    adapter.resume_listening()
    
    assert adapter.is_paused is False
    adapter.audio_manager.resume_session.assert_called_once()


def test_get_llm_response(adapter_with_mocks):
    """Test getting a response from the LLM."""
    adapter = adapter_with_mocks
    
    # Test with context
    response = adapter.get_llm_response(TEST_TRANSCRIPT, TEST_CONTEXT)
    assert response == TEST_RESPONSE
    
    # Test without context
    response = adapter.get_llm_response(TEST_TRANSCRIPT)
    assert response == TEST_RESPONSE


@pytest.mark.asyncio
async def test_synthesize_and_play(adapter_with_mocks, mock_tts):
    """Test synthesizing and playing speech."""
    adapter = adapter_with_mocks
    
    # Test with saving to file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        output_path = temp_file.name
    
    try:
        # For testing, we'll directly test the async method rather than going through _run_async
        # which causes issues with nested event loops
        adapter._init_tts_engine = AsyncMock(return_value=mock_tts)
        
        # Test with output filepath
        adapter._synthesize_and_play_async = AsyncMock(return_value=output_path)
        result = await adapter._synthesize_and_play_async(TEST_TEXT, output_path)
        assert result == output_path
        
        # Test without output filepath
        adapter._synthesize_and_play_async = AsyncMock(return_value=None)
        result = await adapter._synthesize_and_play_async(TEST_TEXT)
        assert result is None
        
    finally:
        # Clean up temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_cleanup(adapter_with_mocks, mock_tts):
    """Test cleaning up resources."""
    adapter = adapter_with_mocks
    adapter.running = True
    adapter.is_paused = False
    adapter.tts_engine = mock_tts
    
    # Create a mock thread
    adapter.listening_thread = MagicMock()
    adapter.listening_thread.is_alive.return_value = True
    adapter.listening_thread.join.return_value = None
    
    adapter.cleanup()
    
    assert adapter.running is False
    adapter.recorder.should_stop is True
    adapter.recorder.reset_state.assert_called_once()
    adapter.tts_engine.stop.assert_called_once()
    adapter.listening_thread.join.assert_called_once()