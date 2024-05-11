import matplotlib.pyplot as plt
import numpy as np

def plot_quantum_state(state):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(state)), np.abs(state)**2)
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title('Quantum State Probability Distribution')
    plt.show()
