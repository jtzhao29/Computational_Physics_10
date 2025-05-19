# /code/ising_experiment_p_distribution.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ising_utils import add_noise, metropolis, cosine_similarity

def simulate_and_plot_distribution(N: int, p_values: List[int], delta: float, beta: float) -> None:
    """
    Simulate for multiple values of p and plot similarity distribution histograms.
    
    Args:
        N (int): Number of spins
        p_values (List[int]): List of reference set sizes
        delta (float): Noise level for generating initial states
        beta (float): Inverse temperature
    """
    plt.figure(figsize=(10, 6))

    for p in p_values:
        v_set = np.random.choice([-1, 1], size=(p, N))
        J = sum(np.outer(v, v) for v in v_set) / N
        np.fill_diagonal(J, 0)

        similarities = []
        for v in v_set:
            s_init = add_noise(v, delta)
            s_final = metropolis(s_init.copy(), J, beta)
            similarities.append(cosine_similarity(s_final, v))

        plt.hist(similarities, bins=30, alpha=0.6, label=f"$p = {p}$")

    plt.xlabel("Cosine similarity $C(s, v)$")
    plt.ylabel("Frequency")
    plt.title("Distribution of $C(s, v)$ for varying $p$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./images/similarity_distribution.png")
    plt.show()
    plt.close()

