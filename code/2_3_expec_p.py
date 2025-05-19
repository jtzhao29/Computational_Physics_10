import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ising_utils import add_noise, metropolis, cosine_similarity
import matplotlib.pyplot as plt

def simulate_and_plot_distribution(N: int, p_values: List[int], delta: float, beta: float,num_samples=10) -> np.array:
    """
    Simulate multiple p values and plot histogram of final similarities C(s, v).
    
    Args:
        N (int): Number of spins
        p_values (List[int]): Different values of p
        delta (float): Noise level
        beta (float): Inverse temperature
    """
    # plt.figure(figsize=(10, 6))
    simi_set=np.zeros((len(p_values),2))
    for p in p_values:
        print(p)
        v_set = np.random.choice([-1, 1], size=(p, N))
        J = sum(np.outer(v, v) for v in v_set) / N
        np.fill_diagonal(J, 0)

        # similarities = []
        for v in v_set:
            C=0
            for _ in range(num_samples):
                s_init = add_noise(v, delta)
                s_final = metropolis(s_init.copy(), J, beta)
                C += cosine_similarity(s_final, v)
            similarities=C/num_samples
        simi_set[p_values.index(p)][0]=p
        simi_set[p_values.index(p)][1]=np.mean(similarities)
    return simi_set
        # print(len(similarities))

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



if __name__ == "__main__":
    N = 2000
    p_values = [274,275,276,277,278,279,280,281,282]
    p_values = [200,210,220,230,240,250,260,270,280,290]
    # p_values = [50,100]

    # p_values = [50,500]
    delta = 0.5
    beta = 1e10

    simu_set=simulate_and_plot_distribution(N, p_values, delta, beta,10)
    print(simu_set)
    print("Similarity distribution plot savedas similarity_distribution.png")
    plt.figure(figsize=(10, 6))
    plt.plot(simu_set[:,0],simu_set[:,1],marker='o',label='Mean of Similarity')
    plt.xlabel("p",fontsize=15)
    plt.ylabel("Mean of Similarity",fontsize=15)
    plt.title(r"Mean of Similarity for different $p$",fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./images/mean_of_similarity_distribution_p={p_values}.png")
    plt.close()