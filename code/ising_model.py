import numpy as np
from typing import Tuple

def cosine_similarity(s: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between spin vectors s and v.
    
    Args:
        s (np.ndarray): Spin vector (±1), shape (N,)
        v (np.ndarray): Spin vector (±1), shape (N,)
    
    Returns:
        float: Cosine similarity C(s, v)
    """
    return np.dot(s, v) / len(s)

def metropolis(s: np.ndarray, J: np.ndarray, beta: float, max_iter: int = 10000) -> np.ndarray:
    """
    Perform Metropolis updates on a spin configuration s.
    
    Args:
        s (np.ndarray): Initial spin vector (±1), shape (N,)
        J (np.ndarray): Coupling matrix, shape (N, N)
        beta (float): Inverse temperature
        max_iter (int): Number of update steps
    
    Returns:
        np.ndarray: Final spin configuration
    """
    N = len(s)
    for _ in range(max_iter):
        i = np.random.randint(N)
        dE = 2 * s[i] * np.dot(J[i], s)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            s[i] *= -1
    return s

def add_noise(v: np.ndarray, delta: float) -> np.ndarray:
    """
    Add noise to spin vector v with flipping probability delta.
    
    Args:
        v (np.ndarray): Spin vector (±1), shape (N,)
        delta (float): Flip probability
    
    Returns:
        np.ndarray: Noisy spin vector
    """
    noise = np.random.choice([-1, 1], size=v.shape)
    mask = np.random.rand(*v.shape) < delta
    return np.where(mask, noise, v)

def simulate_p10_with_noise(N: int = 2000, p: int = 10, delta: float = 0.5, beta: float = 100) -> list:
    """
    Simulate the evolution of noisy initial states and compute similarity.
    
    Args:
        N (int): Number of spins
        p (int): Number of reference states
        delta (float): Noise level
        beta (float): Inverse temperature
    
    Returns:
        List[float]: Final similarities C(s_final, v) for each mu
    """
    v_set = np.random.choice([-1, 1], size=(p, N))
    J = sum(np.outer(v, v) for v in v_set) / N
    np.fill_diagonal(J, 0)

    similarities = []
    for v in v_set:
        s_init = add_noise(v, delta)
        s_final = metropolis(s_init.copy(), J, beta)
        C = cosine_similarity(s_final, v)
        similarities.append(C)
    return similarities


if __name__ == "__main__":
    # Example usage
    N = 1000
    v = np.random.choice([-1, 1], size=N)
    s = np.random.choice([-1, 1], size=N)

    J = np.outer(v, v) / N
    np.fill_diagonal(J, 0)

    print("初始 C(s, v):", cosine_similarity(s, v))
    s_final = metropolis(s.copy(), J, beta=100)
    print("最终 C(s, v):", cosine_similarity(s_final, v))
