"""
EcoSense AI - Audio Augmentation
"""

import random
import librosa
import numpy as np


def add_background_noise(y):
    noise = np.random.randn(len(y))
    noise_factor = random.uniform(0.001, 0.01)
    return (y + noise_factor * noise).astype(np.float32)


def time_stretch_audio(y):
    rate = random.uniform(0.90, 1.10)
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def pitch_shift_audio(y, sr=22050):
    steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=steps
    ).astype(np.float32)


def change_volume(y):
    gain = random.uniform(0.8, 1.2)
    return (y * gain).astype(np.float32)


def random_augmentation(y, sr=22050):
    augmentations = [
        lambda x: add_background_noise(x),
        lambda x: time_stretch_audio(x),
        lambda x: pitch_shift_audio(x, sr),
        lambda x: change_volume(x),
    ]

    augmented = y.copy()

    n = random.choice([1, 2])

    for aug in random.sample(augmentations, n):
        augmented = aug(augmented)

    return augmented