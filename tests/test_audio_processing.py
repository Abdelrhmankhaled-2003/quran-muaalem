import numpy as np

from quran_muaalem.audio_processing import (
    reduce_background_noise,
    trim_trailing_noise,
)


def test_reduce_background_noise_improves_synthetic_recitation_snr():
    sample_rate = 16000
    random = np.random.default_rng(42)
    time = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    envelope = np.zeros_like(time)
    envelope[int(0.5 * sample_rate) : int(3.5 * sample_rate)] = 1
    signal = envelope * (
        0.15 * np.sin(2 * np.pi * 220 * time)
        + 0.08 * np.sin(2 * np.pi * 440 * time)
        + 0.04 * np.sin(2 * np.pi * 660 * time)
    )
    noise = random.normal(0, 0.04, size=time.size).astype(np.float32)
    noise += 0.04 * np.sin(2 * np.pi * 120 * time)
    noisy_audio = signal + noise

    cleaned_audio = reduce_background_noise(noisy_audio, sample_rate)

    input_noise_power = np.mean((noisy_audio - signal) ** 2)
    output_noise_power = np.mean((cleaned_audio - signal) ** 2)
    assert cleaned_audio.shape == noisy_audio.shape
    assert np.isfinite(cleaned_audio).all()
    assert output_noise_power < input_noise_power


def test_reduce_background_noise_uses_long_noise_only_tail():
    sample_rate = 16000
    random = np.random.default_rng(42)
    time = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    signal = np.zeros_like(time)
    speech = slice(int(0.5 * sample_rate), int(3.2 * sample_rate))
    signal[speech] = 0.15 * np.sin(2 * np.pi * 220 * time[speech])
    noise = random.normal(0, 0.01, size=time.size).astype(np.float32)
    noise += 0.03 * np.sin(2 * np.pi * 120 * time)
    noisy_audio = signal + noise

    cleaned_audio = reduce_background_noise(noisy_audio, sample_rate)
    tail = slice(int(3.4 * sample_rate), None)

    assert np.mean(cleaned_audio[tail] ** 2) < np.mean(noisy_audio[tail] ** 2) * 0.5


def test_trim_trailing_noise_preserves_speech_with_context():
    sample_rate = 16000
    random = np.random.default_rng(42)
    time = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    signal = np.zeros_like(time)
    signal[int(0.5 * sample_rate) : int(3.3 * sample_rate)] = 0.15 * np.sin(
        2 * np.pi * 220 * time[int(0.5 * sample_rate) : int(3.3 * sample_rate)]
    )
    noise = random.normal(0, 0.01, size=time.size).astype(np.float32)

    trimmed_audio = trim_trailing_noise(signal + noise, sample_rate)

    assert trimmed_audio.size < time.size
    assert trimmed_audio.size >= int(3.7 * sample_rate)
    assert np.isfinite(trimmed_audio).all()


def test_trim_trailing_noise_keeps_audio_active_through_the_end():
    sample_rate = 16000
    time = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
    signal = 0.15 * np.sin(2 * np.pi * 220 * time)

    trimmed_audio = trim_trailing_noise(signal, sample_rate)

    assert trimmed_audio.shape == signal.shape
