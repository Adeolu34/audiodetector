"""Audio preprocessing and feature extraction utilities."""

import random

import numpy as np
import librosa


def load_audio(path: str, sr: int = 22050) -> tuple:
    """Load an audio file, convert to mono, and resample.

    Returns (waveform, sample_rate) where waveform is a 1-D numpy array.
    """
    waveform, sample_rate = librosa.load(path, sr=sr, mono=True)
    return waveform, sample_rate


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize the waveform to [-1, 1]."""
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return waveform


def trim_silence(waveform: np.ndarray, sr: int, top_db: int = 20) -> np.ndarray:
    """Trim leading and trailing silence from the waveform."""
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute a log-mel spectrogram from a waveform.

    Returns a numpy array of shape (n_mels, T).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


# ---------------------------------------------------------------------------
# Waveform-level augmentation
# ---------------------------------------------------------------------------

def augment_time_shift(waveform: np.ndarray, sr: int, max_shift: float = 0.5) -> np.ndarray:
    """Randomly roll the waveform by up to ±max_shift seconds."""
    max_samples = int(sr * max_shift)
    shift = np.random.randint(-max_samples, max_samples + 1)
    return np.roll(waveform, shift)


def augment_gaussian_noise(waveform: np.ndarray, snr_range: tuple = (15, 30)) -> np.ndarray:
    """Add Gaussian noise at a random SNR within *snr_range* (dB)."""
    snr_db = np.random.uniform(*snr_range)
    signal_power = np.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-10)), len(waveform))
    return waveform + noise


def augment_pitch_shift(waveform: np.ndarray, sr: int, semitone_range: float = 2.0) -> np.ndarray:
    """Pitch-shift the waveform by a random amount within ±semitone_range."""
    n_steps = np.random.uniform(-semitone_range, semitone_range)
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


# ---------------------------------------------------------------------------
# Spectrogram-level augmentation (SpecAugment)
# ---------------------------------------------------------------------------

def spec_augment(
    mel_spec: np.ndarray,
    freq_masks: int = 1,
    freq_width: int = 10,
    time_masks: int = 1,
    time_width: int = 20,
) -> np.ndarray:
    """Apply SpecAugment frequency and time masking.

    Masked regions are filled with the mean value of the spectrogram.
    """
    spec = mel_spec.copy()
    n_mels, n_time = spec.shape
    fill_value = spec.mean()

    for _ in range(freq_masks):
        f = np.random.randint(0, max(freq_width, 1))
        f0 = np.random.randint(0, max(n_mels - f, 1))
        spec[f0:f0 + f, :] = fill_value

    for _ in range(time_masks):
        t = np.random.randint(0, max(time_width, 1))
        t0 = np.random.randint(0, max(n_time - t, 1))
        spec[:, t0:t0 + t] = fill_value

    return spec


def preprocess_file(
    path: str,
    sr: int = 22050,
    duration: float = 5.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    augment: bool = False,
) -> np.ndarray:
    """Full preprocessing pipeline for a single audio file.

    Steps: load -> normalize -> trim silence -> [augment waveform] ->
    pad/truncate -> extract mel spectrogram -> [SpecAugment].

    Args:
        augment: If True, apply waveform and spectrogram augmentations
                 (training only). Defaults to False.

    Returns a log-mel spectrogram as a numpy array of shape (n_mels, T).
    """
    waveform, sr = load_audio(path, sr=sr)
    waveform = normalize_audio(waveform)
    waveform = trim_silence(waveform, sr)

    # Waveform augmentations (before pad/truncate)
    if augment:
        waveform = augment_time_shift(waveform, sr)
        waveform = augment_gaussian_noise(waveform)
        if random.random() < 0.5:
            waveform = augment_pitch_shift(waveform, sr)

    # Pad or truncate to fixed duration
    target_length = int(sr * duration)
    if len(waveform) < target_length:
        pad_amount = target_length - len(waveform)
        waveform = np.pad(waveform, (0, pad_amount), mode="constant")
    else:
        waveform = waveform[:target_length]

    mel_spec = extract_mel_spectrogram(
        waveform, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    # Spectrogram augmentation (SpecAugment)
    if augment:
        mel_spec = spec_augment(
            mel_spec,
            freq_masks=random.randint(1, 2),
            freq_width=10,
            time_masks=random.randint(1, 2),
            time_width=20,
        )

    return mel_spec
