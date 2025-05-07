# backend/src/music_gen.py
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# load once
_model = MusicGen.get_pretrained('facebook/musicgen-small') 
_model.set_generation_params(duration=10)

def generate_music(emotion: str, genre: str) -> str:
    descriptions = [f"{emotion} {genre}"]

    # pure T2A (no melody) call:
    wavs = _model.generate(descriptions)

    # prepare output folder
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{emotion}_{genre}"
    out_path = os.path.join(out_dir, filename)

    # write the first clip
    audio_write(out_path, wavs[0].cpu(), _model.sample_rate, strategy="loudness")
    return f"{filename}.wav"
