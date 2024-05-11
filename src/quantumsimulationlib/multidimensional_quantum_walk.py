import numpy as np

class MultiDimensionalQuantumWalk:
    def __init__(self, dimensions, size, start_position, coin_type='Hadamard'):
        self.dimensions = dimensions
        self.size = size
        self.grid_shape = (size,) * dimensions
        self.position_states = np.zeros((2,) + self.grid_shape, dtype=complex)
        self.start_position = start_position
        self.coin_type = coin_type

        # Initialize walker position
        self.position_states[(0,) + start_position] = 1 / np.sqrt(2)
        self.position_states[(1,) + start_position] = 1 / np.sqrt(2)

    def hadamard_coin(self, state):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        return np.tensordot(H, state, axes=[1, 0])

    def shift(self):
        new_state = np.zeros_like(self.position_states)
        for dimension in range(self.dimensions):
            for direction in [0, 1]:
                axis = direction + 1
                roll = 1 if direction == 0 else -1
                slices = [slice(None)] * (self.dimensions + 1)
                slices[axis] = slice(None, -1) if direction == 0 else slice(1, None)
                target_slices = slices.copy()
                target_slices[axis] = slice(1, None) if direction == 0 else slice(None, -1)
                new_state[tuple(slices)] += np.roll(self.position_states[tuple(target_slices)], roll, axis=axis)
        self.position_states = new_state

    def step(self):
        self.position_states = self.hadamard_coin(self.position_states)
        self.shift()

    def measure(self):
        probability_distribution = np.sum(np.abs(self.position_states)**2, axis=0)
        return probability_distribution

