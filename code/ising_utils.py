# /code/ising_utils.py

import numpy as np
from typing import List
from numba import jit

@jit
def cosine_similarity(s: np.ndarray, v: np.ndarray) -> float:
    s = s.astype(np.float64)
    v = v.astype(np.float64)
    return np.dot(s, v) / len(s)



@jit
def add_noise(v: np.ndarray, delta: float) -> np.ndarray:
    noise = np.random.choice(np.array([-1, 1]), size=v.shape)
    mask = np.random.rand(*v.shape) < delta
    return np.where(mask, noise, v)

@jit
def calculate_enengy(J:np.array,s:np.array)->float:
    """
    输入：J,s
    输出：E

    """
    
    # $$H = -\frac{1}{2} \sum_{i \neq j} J_{ij} s_i s_j,$$
    return -0.5 * s @ J @ s


# 测试E
if __name__ == "__main__":
    J = np.array([[0, 1], [1, 0]])
    s = np.array([1, -1])
    E = calculate_enengy(J, s)
    print(J)
    print(E)



@jit
def metropolis(s: np.ndarray, J: np.ndarray, beta: float, max_iter: int = 10000) -> np.ndarray:
    N = len(s)
    s = s.astype(np.float64)
    for _ in range(max_iter):
        i = np.random.randint(N)
        dE = 2 * s[i] * np.dot(J[i], s)  # ΔE = 2 s_i ∑_j J_ij s_j
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            s[i] *= -1
    return s


@jit
def metropolis(s: np.ndarray, J: np.ndarray, beta: float, max_iter: int = 10000) -> np.ndarray:
    N = len(s)
    s = s.astype(np.float64)
    for _ in range(max_iter):
        i = np.random.randint(N)
        dE = 2 * s[i] * np.dot(J[i], s)  # ΔE = 2 s_i ∑_j J_ij s_j
        # if dE < 0 or np.random.rand() < np.exp(-beta * dE):
        if dE < 0 or dE==0 :
            s[i] *= -1
    return s
@jit
def generate_random_spin_configuration(N: int) -> np.ndarray:
    return np.random.choice([-1, 1], size=N)

@jit
def simulate_p10_with_noise(N: int = 2000, p: int = 10, delta: float = 0.5, beta: float = 100) -> List[float]:
    s = generate_random_spin_configuration(N)
    J= np.zeros((N, N))

@jit
def calculate_similarity(s: np.ndarray, v: np.ndarray) -> float:
    """
    输入：s,v
    输出：相似度(不带绝对值)
    """
    return np.dot(s, v) / np.linalg.norm(s) / np.linalg.norm(v)
