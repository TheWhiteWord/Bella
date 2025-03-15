from typing import List, Dict, Optional
from dataclasses import dataclass
from extended_features.multi_speaker import MultiSpeakerGenerator

@dataclass
class ConversationTurn:
    role: str
    content: str
    speaker_id: int

class ConversationManager:
    def __init__(self, generator: MultiSpeakerGenerator):
        self.generator = generator
        self.conversation_history: List[ConversationTurn] = []
        self.special_commands = {
            "$CLEAR$": self._handle_clear,
            "$SWAP$": self._handle_swap,
            "$BACK$": self._handle_back
        }

    def process_input(self, text: str, output_filename: str = "output.wav") -> Optional[str]:
        """Process input text, handling special commands and generating audio."""
        if text in self.special_commands:
            return self.special_commands[text]()
        
        # Generate audio and add to conversation history
        output_file = self.generator.generate_multi_speaker(text, output_filename)
        
        # Add each speaker's part to conversation history
        segments = self.generator.process_multi_speaker_text(text)
        for segment_text, speaker_id in segments:
            self.conversation_history.append(
                ConversationTurn(
                    role="user" if speaker_id == self.generator.current_speaker else "assistant",
                    content=segment_text,
                    speaker_id=speaker_id
                )
            )
        
        return output_file

    def _handle_clear(self) -> None:
        """Handle $CLEAR$ command."""
        self.generator.clear_context()
        self.conversation_history.clear()
        return None

    def _handle_swap(self) -> None:
        """Handle $SWAP$ command."""
        self.generator.increment_speaker()
        return None

    def _handle_back(self) -> None:
        """Handle $BACK$ command."""
        self.generator.decrement_speaker()
        return None

    def get_conversation_history(self) -> List[ConversationTurn]:
        """Get the conversation history."""
        return self.conversation_history

    def export_conversation(self) -> List[Dict]:
        """Export conversation in a format compatible with common chat APIs."""
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "speaker_id": turn.speaker_id
            }
            for turn in self.conversation_history
        ]