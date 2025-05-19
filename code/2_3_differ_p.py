# /code/ising_experiment_p_distribution.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ising_utils import add_noise, metropolis, cosine_similarity
import matplotlib.pyplot as plt

def simulate_and_plot_distribution(N: int, p_values: List[int], delta: float, beta: float,num_samples=10) -> None:
    """
    Simulate multiple p values and plot histogram of final similarities C(s, v).
    
    Args:
        N (int): Number of spins
        p_values (List[int]): Different values of p
        delta (float): Noise level
        beta (float): Inverse temperature
    """
    plt.figure(figsize=(10, 6))
    for p in p_values:
        print(p)
        v_set = np.random.choice([-1, 1], size=(p, N))
        J = sum(np.outer(v, v) for v in v_set) / N
        np.fill_diagonal(J, 0)

        similarities = []
        for v in v_set:
            C=0
            for _ in range(num_samples):
                s_init = add_noise(v, delta)
                s_final = metropolis(s_init.copy(), J, beta)
                C += cosine_similarity(s_final, v)
            similarities.append(C/num_samples)
        print(len(similarities))

        # plt.hist(similarities, bins=30, alpha=0.6, label=f"p = {p}",density=True)
        counts, bins = np.histogram(similarities, bins=30)
        counts = counts / np.sum(counts)  # 归一化为概率
        plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.6, label=f"p = {p}")


    plt.xlabel("similarity $C(s, v)$",fontsize=15)
    plt.ylabel("Frequency",fontsize=15)
    plt.title("Similarity distribution for different $p$",fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./images/similarity_distribution_p={p_values}_2.png")
    plt.close()


if __name__ == "__main__":
    N = 2000
    p_values = [50,100,150,200,250,300,350,400,450,500]
    p_values = [240,276,285]
    # p_values = [50,500]
    delta = 0.5
    beta = 1e10

    simulate_and_plot_distribution(N, p_values, delta, beta,15)
    print("Similarity distribution plot savedas similarity_distribution.png")