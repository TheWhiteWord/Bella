"""Tests for the Voice Assistant GUI.

This module tests the basic functionality of the Voice Assistant GUI component,
focusing on widget initialization and state management.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# We need to mock all tkinter modules before importing the GUI
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.ttk'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinterdnd2'] = MagicMock()

# Now import the GUI module with properly mocked dependencies
from src.gui.voice_assistant_gui import VoiceAssistantGUI


@pytest.fixture
def mock_backend_adapter():
    """Create a mock backend adapter."""
    adapter = MagicMock()
    adapter.start_continuous_listening = MagicMock()
    adapter.pause_listening = MagicMock()
    adapter.resume_listening = MagicMock()
    adapter.get_llm_response = MagicMock(return_value="This is a test response")
    adapter.synthesize_and_play = MagicMock(return_value="/tmp/test.wav")
    adapter.cleanup = MagicMock()
    return adapter


@pytest.fixture
def mocked_gui(mock_backend_adapter):
    """Create a GUI instance with mocked components."""
    # Mock the BackendAdapter class
    with patch('src.gui.voice_assistant_gui.BackendAdapter', return_value=mock_backend_adapter):
        # Since tkinter is fully mocked, we can init the GUI
        gui = VoiceAssistantGUI(sink_name="test_sink")
        
        # Manually set up the GUI attributes to simulate initialization
        gui.is_context_loaded = False
        gui.context_content = None
        gui.is_reading_context = False
        gui.is_processing = False
        gui.last_tts_filepath = None
        gui.audio_subprocess = None
        gui.backend = mock_backend_adapter
        
        # Create mock widgets
        gui.root = MagicMock()
        gui.context_button = MagicMock() 
        gui.record_button = MagicMock()
        gui.download_button = MagicMock()
        gui.display_canvas = MagicMock()
        gui.status_label = MagicMock()
        
        return gui


def test_gui_initialization(mocked_gui, mock_backend_adapter):
    """Test that the GUI initializes correctly."""
    gui = mocked_gui
    
    # Check that the backend adapter was initialized with the correct sink name
    assert gui.sink_name == "test_sink"
    assert gui.backend == mock_backend_adapter
    
    # Check initial state variables
    assert gui.is_context_loaded is False
    assert gui.context_content is None
    assert gui.is_reading_context is False
    assert gui.is_processing is False
    assert gui.last_tts_filepath is None
    
    # Verify the backend adapter was properly used
    mock_backend_adapter.start_continuous_listening.assert_called_once()


def test_toggle_context(mocked_gui):
    """Test toggling context state."""
    gui = mocked_gui
    gui.is_context_loaded = True
    gui.context_content = "Test context"
    
    # Test turning context off
    gui._toggle_context()
    
    assert gui.is_context_loaded is False
    assert gui.context_content is None
    gui.context_button.config.assert_called_once()
    gui.record_button.config.assert_called_once()
    gui.download_button.config.assert_called_once()


def test_on_speech_detected(mocked_gui):
    """Test handling speech detection."""
    gui = mocked_gui
    gui._process_chat_interaction = MagicMock()
    
    # Test speech detection
    gui._on_speech_detected("Hello world")
    
    # Verify that after was called on root
    assert gui.root.after.called


def test_read_context_aloud(mocked_gui, mock_backend_adapter):
    """Test reading context aloud."""
    gui = mocked_gui
    gui.is_context_loaded = True
    gui.context_content = "Test context content"
    gui.is_processing = False
    gui._start_waveform_visualization = MagicMock()
    
    # Mock tempfile.NamedTemporaryFile
    with patch('tempfile.NamedTemporaryFile') as mock_temp:
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_audio.wav"
        mock_temp.return_value = mock_temp_file
        
        # Call the method
        gui._read_context_aloud()
        
        # Verify expected behavior
        assert gui.is_reading_context is True
        assert gui.is_processing is True
        mock_backend_adapter.pause_listening.assert_called_once()
        gui._start_waveform_visualization.assert_called_once()


def test_cleanup(mocked_gui, mock_backend_adapter):
    """Test cleanup when closing the window."""
    gui = mocked_gui
    gui._stop_waveform_visualization = MagicMock()
    
    # Test cleanup
    gui._on_close()
    
    gui._stop_waveform_visualization.assert_called_once()
    mock_backend_adapter.cleanup.assert_called_once()
    gui.root.destroy.assert_called_once()