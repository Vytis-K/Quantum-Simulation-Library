import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Define the custom coin operation
def stochastic_coin(pos_state):
    # Randomly choose between Hadamard and Grover coin
    if random.random() < 0.5:
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    else:
        G = 2 * np.full((2, 2), 0.5) - np.eye(2)
    return np.dot(H if random.random() < 0.5 else G, pos_state)

# Simulation function for running the quantum walk
def simulate_quantum_walk(num_positions, num_walks):
    results = []
    for _ in range(num_walks):
        qw = QuantumWalk(num_positions, num_positions // 2, coin_operation=stochastic_coin)
        for _ in range(100):  # Number of steps in each walk
            qw.step()
        results.append(qw.measure())
    return results

# Visualize the results
def visualize_results(results):
    all_final_positions = [result.argmax() for result in results]  # Get the peak position for each walk
    plt.hist(all_final_positions, bins=50, alpha=0.75)
    plt.title('Distribution of Quantum Walk Final Positions')
    plt.xlabel('Final Position')
    plt.ylabel('Frequency')
    plt.show()

# Parameters
num_positions = 100  # Number of positions in the quantum walk
num_walks = 1000     # Number of walks to simulate

# Run the simulation and visualization
results = simulate_quantum_walk(num_positions, num_walks)
visualize_results(results)
