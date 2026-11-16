import noisereduce
import numpy as np

_FRAME_DURATION_SECONDS = 0.1
_NOISE_ACTIVITY_MARGIN_DB = 4.0
_MAX_NOISE_TAIL_VARIATION = 0.15
_MIN_TRAILING_NOISE_SECONDS = 0.55
_TRAILING_AUDIO_PADDING_SECONDS = 0.45


def reduce_background_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("Noise reduction expects mono audio.")
    if audio.size == 0:
        return audio

    frame_size = max(1, round(sample_rate * _FRAME_DURATION_SECONDS))
    frames = [
        audio[start : start + frame_size]
        for start in range(0, audio.size, frame_size)
    ]
    frame_levels = np.array(
        [np.sqrt(np.mean(frame**2)) for frame in frames], dtype=np.float32
    )
    quiet_threshold = np.quantile(frame_levels, 0.2)
    quiet_frames = [
        frame for frame, level in zip(frames, frame_levels) if level <= quiet_threshold
    ]
    speech_end = _speech_end_from_levels(frame_levels, frame_size, audio.size)
    has_noise_tail = (
        speech_end is not None
        and audio.size - speech_end > sample_rate * _MIN_TRAILING_NOISE_SECONDS
        and _has_stable_noise_tail(frame_levels, frame_size, speech_end)
    )
    noise_profile = (
        audio[speech_end:]
        if has_noise_tail and speech_end is not None
        else np.concatenate(quiet_frames)
    )

    if noise_profile.size >= sample_rate * 0.25:
        return noisereduce.reduce_noise(
            y=audio,
            y_noise=noise_profile,
            sr=sample_rate,
            stationary=True,
            n_fft=512,
            n_std_thresh_stationary=0.5,
            prop_decrease=1.0 if has_noise_tail else 0.8,
            freq_mask_smooth_hz=250,
            time_mask_smooth_ms=200,
        )

    return noisereduce.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=False,
        n_fft=512,
        prop_decrease=0.8,
    )


def _speech_end_from_levels(
    frame_levels: np.ndarray, frame_size: int, audio_size: int
) -> int | None:
    if frame_levels.size == 0:
        return None

    quiet_frames = frame_levels <= np.quantile(frame_levels, 0.2)
    noise_floor = np.median(frame_levels[quiet_frames])
    activity_threshold = noise_floor * 10 ** (_NOISE_ACTIVITY_MARGIN_DB / 20)
    active_frames = np.flatnonzero(frame_levels > activity_threshold)
    if active_frames.size == 0:
        return None

    return min((int(active_frames[-1]) + 1) * frame_size, audio_size)


def _has_stable_noise_tail(
    frame_levels: np.ndarray, frame_size: int, speech_end: int
) -> bool:
    first_tail_frame = (speech_end + frame_size - 1) // frame_size
    tail_levels = frame_levels[first_tail_frame:]
    if tail_levels.size < 3:
        return False

    mean_level = np.mean(tail_levels)
    if mean_level <= 0:
        return False

    return np.std(tail_levels) / mean_level <= _MAX_NOISE_TAIL_VARIATION


def trim_trailing_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("Trailing-noise trimming expects mono audio.")
    if audio.size == 0:
        return audio

    frame_size = max(1, round(sample_rate * _FRAME_DURATION_SECONDS))
    frames = [
        audio[start : start + frame_size]
        for start in range(0, audio.size, frame_size)
    ]
    frame_levels = np.array(
        [np.sqrt(np.mean(frame**2)) for frame in frames], dtype=np.float32
    )
    speech_end = _speech_end_from_levels(frame_levels, frame_size, audio.size)
    if speech_end is None:
        return audio

    trailing_noise = audio.size - speech_end
    if (
        trailing_noise <= sample_rate * _MIN_TRAILING_NOISE_SECONDS
        or not _has_stable_noise_tail(frame_levels, frame_size, speech_end)
    ):
        return audio

    trim_end = speech_end + round(sample_rate * _TRAILING_AUDIO_PADDING_SECONDS)
    return audio[: min(trim_end, audio.size)]
