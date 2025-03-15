from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment, CACHE_DIR
import torchaudio
import traceback
import os
import shlex
from subprocess import run

class MultiSpeakerGenerator:
    def __init__(self):
        print("Loading model...")
        self.model_path = os.path.join(CACHE_DIR, "ckpt.pt")
        self.generator = load_csm_1b(self.model_path, "cuda")
        self.current_speaker = 0
        self.context_segments = []
        self.conversation_history = []
        print("Model loaded successfully!")

    def load_reference_audio(self, reference_path="reference.wav", reference_text="This is Sesame. I say hi. And pwee. And more!"):
        if os.path.exists(reference_path):
            print(f"Using {reference_path} as reference audio")
            reference_audio = self._load_audio(reference_path)
            self.context_segments = [
                Segment(text=reference_text, speaker=0, audio=reference_audio)
            ]
            self.conversation_history.append({"role": "assistant", "content": reference_text})
            print("Reference audio loaded and added to context")
            return True
        return False

    def _load_audio(self, audio_path):
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=self.generator.sample_rate
        )
        return audio_tensor

    def clear_context(self):
        self.context_segments = []
        self.conversation_history = []
        return "Context cleared."

    def increment_speaker(self):
        self.current_speaker += 1
        return f"Speaker incremented to {self.current_speaker}"

    def decrement_speaker(self):
        self.current_speaker -= 1
        return f"Speaker decremented to {self.current_speaker}"

    def process_multi_speaker_text(self, text):
        """Process text with || separators and return segments with speaker IDs"""
        if "||" not in text:
            return [(text, self.current_speaker)]
        
        segments = []
        parts = text.split("||")
        
        for i, part in enumerate(parts):
            if i == 0:
                segments.append((part, self.current_speaker))
                continue
                
            try:
                if part[-2] == '-':
                    speaker_offset = int(part[-2:])
                    clean_text = part[:-2]
                else:
                    speaker_offset = int(part[-1])
                    clean_text = part[:-1]
            except:
                speaker_offset = i
                clean_text = part
                
            segments.append((clean_text, self.current_speaker + speaker_offset))
            
        return segments

    def generate_multi_speaker(self, text, output_filename="outputCombined.wav"):
        """Generate audio for multiple speakers and combine them"""
        segments = self.process_multi_speaker_text(text)
        temp_files = []
        
        for i, (segment_text, speaker_id) in enumerate(segments):
            print(f"Generating audio for: '{segment_text}' (Speaker {speaker_id})")
            
            audio = self.generator.generate(
                text=segment_text,
                speaker=speaker_id,
                context=self.context_segments,
                max_audio_length_ms=25_000,
            )
            
            temp_filename = f"temp_{i}.wav"
            temp_files.append(temp_filename)
            torchaudio.save(temp_filename, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
            
            self.context_segments.append(
                Segment(text=segment_text, speaker=speaker_id, audio=audio)
            )
            
            if len(self.context_segments) > 5:
                self.context_segments = self.context_segments[-5:]
            
            self.conversation_history.append({"role": "user", "content": segment_text})

        # Combine audio files using ffmpeg
        if len(temp_files) > 1:
            filter_complex = ''.join(f'[{i}:0]' for i in range(len(temp_files))) + \
                           f'concat=n={len(temp_files)}:v=0:a=1[out]'
            
            input_files = ' '.join(f'-i {f}' for f in temp_files)
            cmd = f"ffmpeg {input_files} -filter_complex '{filter_complex}' -map '[out]' {output_filename}"
            run(shlex.split(cmd))
            
            # Cleanup temp files
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
        else:
            # If only one segment, just rename the temp file
            os.rename(temp_files[0], output_filename)
            
        return output_filename