# /code/ising_utils.py

import numpy as np
from typing import List

def cosine_similarity(s: np.ndarray, v: np.ndarray) -> float:
    return np.dot(s, v) / len(s)

def add_noise(v: np.ndarray, delta: float) -> np.ndarray:
    noise = np.random.choice([-1, 1], size=v.shape)
    mask = np.random.rand(*v.shape) < delta
    return np.where(mask, noise, v)

def metropolis(s: np.ndarray, J: np.ndarray, beta: float, max_iter: int = 10000) -> np.ndarray:
    N = len(s)
    for _ in range(max_iter):
        i = np.random.randint(N)
        dE = 2 * s[i] * np.dot(J[i], s)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            s[i] *= -1
    return s
