import librosa
import sounddevice as sd

musics = [
    "Guns N' Roses - Civil War.mp3",
    "Star Wars- The Imperial March.mp3",
    "Daft Punk - Harder, Better, Faster, Stronger.mp3"
]

def play_music(file_path, audio_duration=10, audio_start=0.8):
    y, sr = librosa.load(file_path, sr=None, mono=False)

    n_samples = y.shape[1]

    start_sample = int(n_samples * audio_start)
    end_sample = int(start_sample + audio_duration * sr)
    end_sample = min(end_sample, n_samples)

    snippet = y[:, start_sample:end_sample]

    sd.play(snippet.T, sr)
    sd.wait()