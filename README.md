# CSM-Multi

A focused implementation of CSM-1B for creating a single-voice chat companion, with additional multi-speaker capabilities preserved for future development.

## Primary Focus
This project is primarily designed to create a chat companion with a consistent, high-quality cloned voice. While it includes multi-speaker capabilities from the original implementation, the main use case is single-voice interaction.

For detailed documentation including both primary features and future possibilities, see `DOCUMENTATION.md`.

## Core Features
- Single voice cloning with persistent characteristics
- Optimized for continuous conversation
- Memory-efficient context management
- High-quality voice reproduction

## Commands
* `$CLEAR$` - Clear the context. Essential for managing memory during long conversations.
* `$SWAP$` - (Reserved for future multi-speaker implementation)
* `$BACK$` - (Reserved for future multi-speaker implementation)

## Extended Features
> Note: These features are maintained but not part of the primary implementation.

### Diffe
**diffe.py** Modified script from issue [61](https://github.com/SesameAILabs/csm/issues/61) from the original Sesame repo. Thanks to [@asiff00](https://github.com/asiff00) for the original code, there is now a way to, while keeping the models loaded, enter multiple text prompts without waiting for a reload each time. I've also added multi speaker support using speaker offset text splitting. simply separate each speaker's text with `||` and add a number to the end of the text that will be used as the offset (else a default offset will be applied). It also included a simple reference audio input capability (needs to be mono, not stereo), not as sophisticated as [csm-voice-cloning](https://github.com/isaiahbjork/csm-voice-cloning) repo's voice-clone.py, of which I've also added a modification for. (Thanks to that repo's author for the original, most use instructions can be found there, but new things will be discussed here).

The maximum amount of "multi-speakers" for now is 4, but this can be experimented with in the code. I've not yet come up with a better solution to ffmpeg concat argument increases per gen.\

### Offsets usage:
Example input:
`Yeah, they're real cool.||I see that Yala! How rare are they?||I dunno, they're a pwii.0`

The `||`s are the separators for different generations in one go (4 max) and the option `0` is the offset to the speaker, which defaults to +1 per `||`. In this case, the first speaker will speak, then the second, then the first again.

### Voice-Clone

Thanks to the efforts in this repo: [csm-voice-cloning](https://github.com/isaiahbjork/csm-voice-cloning) a "better than simple reference" voice cloning via Sesame exists. See voice `cloning.md` from the orignial repo from more. New features added are the same multispeaker functionality and commands from diffe.

To use the 4096 (or other context sizes) rename `models.py` to `models.py.old` or some such and rename `models-cloning.py` to `models.py` and follow the instructions in the original csm-voice-cloning repo on how to change that and the `voice_clone.py` file for more context.

## Prerequisites

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Usage

Generate a sentence

```python
from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, device)
audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

```python
speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.
