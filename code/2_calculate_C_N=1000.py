import numpy as np
import matplotlib.pyplot as plt
from ising_utils import metropolis, calculate_similarity, add_noise,generate_random_spin_configuration

if __name__ == "__main__":
    v  = generate_random_spin_configuration(1000)
    s = generate_random_spin_configuration(1000)
    similarity = abs(calculate_similarity(s, v))
    print("similarity of s and v when N=1000: ",similarity)