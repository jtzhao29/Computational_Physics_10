import numpy as np
from typing import List
from ising_model import add_noise, metropolis, cosine_similarity
from IPython.display import display, Math




def simulate_p10_with_noise(N: int = 2000, p: int = 10, delta: float = 0.5, beta: float = 1e5) -> List[float]:
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
    i=1
    for v in v_set:
        s_init = add_noise(v, delta)
        s_final = metropolis(s_init.copy(), J, beta)
        C = cosine_similarity(s_final, v)
        similarities.append(C)
        # print(r"$similarity of s_{i} and v_{i}$",C)
        # display(Math(r"similarity\ of\ s_{%d}\ and\ v_{%d} = %.3f" % (i, i, C)))
    return similarities


if __name__ == "__main__":
    N = 2000
    p = 10
    delta = 0.5
    beta = 100
    similarities = simulate_p10_with_noise(N, p, delta, beta)
    # print("average of similarities: ",similarities)
    for i, C in enumerate(similarities):
        print(f"for {i}th, the similarities is: ",C)
    print("average of similarities: ",np.sum(similarities)/10)
