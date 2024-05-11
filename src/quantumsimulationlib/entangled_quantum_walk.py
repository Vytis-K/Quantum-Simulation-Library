import numpy as np
import networkx as nx

class EntangledQuantumWalk:
    def __init__(self, num_positions, num_particles, dimension=1, topology='line', coin_type='Hadamard'):
        self.dimension = dimension
        self.topology = topology
        self.coin_type = coin_type
        self.num_particles = num_particles

        if topology == 'network':
            self.graph = nx.random_regular_graph(3, num_positions)
            self.position_states = np.zeros((2 ** num_particles, nx.number_of_nodes(self.graph)), dtype=complex)
        else:
            shape = (2 ** num_particles, *([num_positions] * dimension))
            self.position_states = np.zeros(shape, dtype=complex)

        for i in range(num_particles):
            self.position_states[1 << i, i] = 1 / np.sqrt(num_particles)

    def apply_coin(self):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        for idx in range(2 ** self.num_particles):
            #self.position_states[idx] = np.kron(H, self.position_states[idx])
            if self.dimension > 1 or self.topology == 'network':
                # Apply coin flip for each node in network
                for node in range(self.position_states.shape[1]):
                    self.position_states[idx, node] = np.dot(H, self.position_states[idx, node])
            else:
                # Apply coin flip across all positions for each state configuration
                for pos in np.ndindex(*self.position_states[idx].shape):
                    self.position_states[idx][pos] = np.dot(H, self.position_states[idx][pos])
                pass

    def shift(self):
        new_state = np.zeros_like(self.position_states, dtype=complex)
        if self.topology == 'line' or self.topology == 'grid':
            # Apply shifts along each dimension
            for axis in range(1, 1 + self.dimension):
                new_state += np.roll(self.position_states, shift=1, axis=axis)
                new_state += np.roll(self.position_states, shift=-1, axis=axis)
            self.position_states = new_state / (2 * self.dimension)
        elif self.topology == 'network':
            A = nx.adjacency_matrix(self.graph).toarray()
            for idx in range(2 ** self.num_particles):
                new_state[idx] = A.dot(self.position_states[idx])
            self.position_states = new_state

    def apply_decoherence(self, rate=0.01):
        for idx in range(2 ** self.num_particles):
            noise = (np.random.rand(*self.position_states[idx].shape) < rate) * np.random.normal(loc=0.0, scale=1.0, size=self.position_states[idx].shape)
            self.position_states[idx] += noise
            norm = np.sum(np.abs(self.position_states[idx])**2)
            self.position_states[idx] /= np.sqrt(norm)

    def step(self):
        self.apply_coin()
        self.shift()
        self.apply_decoherence()

    def measure(self):
        probabilities = np.sum(np.abs(self.position_states)**2, axis=0)
        return probabilities
