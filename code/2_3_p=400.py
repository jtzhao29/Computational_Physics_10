import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ising_utils import add_noise, metropolis, cosine_similarity
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.stats import norm

# from 2_3_differ_p import simulate_and_plot_distribution


def simulate_and_plot_distribution(N: int, p:int, delta: float, beta: float,num_samples=10) -> np.array:
    """
    Simulate multiple p values and plot histogram of final similarities C(s, v).
    
    Args:
        N (int): Number of spins
        p_values (List[int]): Different values of p
        delta (float): Noise level
        beta (float): Inverse temperature
    """
    # plt.figure(figsize=(10, 6))
    # for p in p_values:
    print(p)
    v_set = np.random.choice(np.array([1,-1]), size=(p, N))
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
    # counts, bins = np.histogram(similarities, bins=30)
    # counts = counts / np.sum(counts)  # 归一化为概率
    # plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.6, label=f"p = {p}")


    # plt.xlabel("similarity $C(s, v)$",fontsize=15)
    # plt.ylabel("Frequency",fontsize=15)
    # plt.title("Similarity distribution for different $p$",fontsize=15)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"./images/similarity_distribution_p={p_values}_2.png")
    # plt.close()
    return similarities

# if __name__ == "__main__":
#     N = 2000
#     p_values = 400
#     delta = 0.5
#     beta = 100
#     num_samples = 1
#     simu_sets = simulate_and_plot_distribution(N, p_values, delta, beta, num_samples)
#     # 拟合simu_sets
#     simu_sets = np.array(simu_sets)
#     df = pd.DataFrame(simu_sets)
#     df.to_csv("./data/similarity_distribution_p400.csv", index=False)
#     # simu_sets.to_csv("./data/similarity_distribution_p400.csv")
#     plt.figure(figsize=(10, 6))
#     # counts, bins = np.histogram(simu_sets , bins=30)
#     # counts = counts / np.sum(counts)  # 归一化为概率
#     # plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.6, label=f"p = {p_values}")

#     aaa = np.zeros((len(bins)-1,2))
#     aaa[:,0] = bins[:-1]
#     aaa[:,1] = counts
#     # 高斯拟合
#     mu, std = norm.fit(aaa)
#     x = np.linspace(bins[0], bins[-1], 300)
#     pdf = norm.pdf(x, mu, std)
#     plt.plot(x, pdf, 'r--', linewidth=2, label=f"Gaussian Fit\n$\\mu={mu:.3f}, \\sigma={std:.3f}$")

#     # 图形标签
#     plt.xlabel("Similarity $C(s, v)$", fontsize=15)
#     plt.ylabel("Probability Density", fontsize=15)
#     plt.title("Similarity Distribution with Gaussian Fit", fontsize=16)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("./images/similarity_distribution_fit_p400.png")
#     plt.show()


if __name__ == "__main__":
    N = 2000
    p = 400
    delta = 0.5
    beta = 100
    num_samples = 25

    # 模拟
    simu_sets = simulate_and_plot_distribution(N, p, delta, beta, num_samples)
    simu_sets = np.array(simu_sets)

    # 直方图
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(simu_sets, bins=30, density=True, alpha=0.6, label=f"$p = {p}$")

    # 高斯拟合
    mu, std = norm.fit(simu_sets)
    x = np.linspace(bins[0], bins[-1], 300)
    pdf = norm.pdf(x, mu, std)
    plt.plot(x, pdf, 'r--', linewidth=2, label=f"Gaussian Fit\n$\\mu={mu:.3f}, \\sigma={std:.3f}$")

    # 图形标签
    plt.xlabel("Similarity $C(s, v)$", fontsize=15)
    plt.ylabel("Probability Density", fontsize=15)
    plt.title("Similarity Distribution with Gaussian Fit", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./images/similarity_distribution_fit_p400.png")
    plt.show()
