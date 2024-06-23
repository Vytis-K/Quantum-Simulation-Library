import numpy as np
from quantum_walk import QuantumWalk

def custom_coin(position_state):
    # Example custom coin: X gate (Pauli-X)
    C = np.array([[0, 1], [1, 0]])
    return np.dot(C, position_state)

def main():
    # Create a QuantumWalk instance with a custom coin
    quantum_walk = QuantumWalk(num_positions=10, start_position=5, coin_operation=custom_coin)
    quantum_walk.step()  # Perform a step in the quantum walk
    print("Probability distribution after one step:", quantum_walk.measure())

if __name__ == "__main__":
    main()
