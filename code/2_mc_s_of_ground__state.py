import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ising_utils import add_noise, metropolis, calculate_similarity



if __name__ == "__main__":
    m=0
    for  _ in range(10):
        # Example usage
        N = 1000
        v = np.random.choice([-1, 1], size=N)
        s = np.random.choice([-1, 1], size=N)


        # v = np.array([1, 2, 3])
        J = np.outer(v, v) /N
        np.fill_diagonal(J, 0)
        # print("J:", J)

        print("初始 C(s, v):", calculate_similarity(s, v))
        s_final = metropolis(s.copy(), J, beta=1e20)
        # print("final s: ",s)
        # 精确到小数点后6位
        print("最终 C(s, v):", calculate_similarity(s_final, v))
        m+=calculate_similarity(s_final, v)
    print("平均 C(s, v):", m/1000)
    