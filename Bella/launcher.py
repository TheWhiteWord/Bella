"""Launcher script for Bella Voice Assistant.

This script serves as the entry point for launching Bella in either CLI or GUI mode.
It preserves all original functionality while adding GUI support.

Usage:
    python launcher.py             # Launches CLI mode (default)
    python launcher.py --gui       # Launches GUI mode
    python launcher.py --help      # Shows all available options
"""
import os
import sys
import argparse
import asyncio
from typing import Optional

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Bella Voice Assistant")
    
    # Common arguments for both modes
    parser.add_argument(
        "--sink",
        type=str,
        help="Name of PulseAudio sink to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for responses (CLI mode only)"
    )
    
    # Mode selection
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch in GUI mode"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    return parser.parse_args()

async def start_gui_mode(sink_name: Optional[str] = None) -> None:
    """Launch the voice assistant in GUI mode.
    
    Args:
        sink_name (str, optional): Name of PulseAudio sink to use
    """
    from src.gui.voice_assistant_gui import VoiceAssistantGUI
    
    app = VoiceAssistantGUI(sink_name=sink_name)
    app.run()  # This will block until the GUI is closed

async def start_cli_mode(args: argparse.Namespace) -> None:
    """Launch the voice assistant in CLI mode.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    from main import main_interaction_loop, list_audio_devices
    
    if args.list_devices:
        list_audio_devices()
        return
    
    await main_interaction_loop(args.model, args.sink)

def main() -> None:
    """Main entry point for the launcher."""
    args = parse_arguments()
    
    try:
        if args.gui:
            print("Starting Bella Voice Assistant in GUI mode...")
            asyncio.run(start_gui_mode(args.sink))
        else:
            print("Starting Bella Voice Assistant in CLI mode...")
            asyncio.run(start_cli_mode(args))
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()