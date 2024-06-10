#contains logic of package
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider

class QuantumWalk:
    def __init__(self, num_positions, start_position, coin_operation=None):
        self.num_positions = num_positions
        self.position_state = np.zeros((2, num_positions), dtype=complex)
        self.position_state[0, start_position] = 1

        if coin_operation is None:
            self.coin_operation = self.default_coin_operation
        elif isinstance(coin_operation, np.ndarray):
            self.coin_operation = self.create_matrix_operation(coin_operation)
        elif callable(coin_operation):
            self.coin_operation = coin_operation
        else:
            raise TypeError("coin_operation must be either a numpy array or a callable.")

    def reset(self):
        self.position_state = np.zeros((2, self.num_positions), dtype=complex)
        self.position_state[0, self.initial_position] = 1
        self.update_coin_operation()
        self.path_history = [self.position_state.copy()]

    def set_start_position(self, position):
        self.initial_position = position
        self.reset()

    def create_matrix_operation(self, matrix):
        if matrix.shape != (2, 2):
            raise ValueError("Coin matrix must be 2x2.")
        return lambda pos_state: np.dot(matrix, pos_state)

    def default_coin_operation(self):
        if self.coin_type == 'Hadamard':
            H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            return np.dot(H, self.position_state)
        elif self.coin_type == 'Grover':
            G = 2 * np.full((2, 2), 1/2) - np.eye(2)
            return np.dot(G, self.position_state)
        elif self.coin_type == 'Fourier':
            F = np.array([[1, 1], [1, -1j]]) / np.sqrt(2)
            return np.dot(F, self.position_state)
        else:
            raise ValueError("Unsupported default coin type")

    def apply_coin(self):
        coins = {
            'Hadamard': np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            'Grover': 2 * np.full((2, 2), 0.5) - np.eye(2),
            'Fourier': np.array([[1, 1], [1, -1j]]) / np.sqrt(2)
        }

        coin = coins.get(self.coin_type)
        if coin is None:
            raise ValueError("Unsupported coin type")

        for i in range(self.num_positions):
            self.position_state[:, i] = np.dot(achievable, self.position_state[:, i])

    def apply_decoherence(self, rate=0.01, model='gaussian'):
        if model == 'gaussian':
            noise = np.random.normal(0, rate, self.position_state.shape) + 1j * np.random.normal(0, rate, self.position_state.shape)
            self.position_state += noise
        elif model == 'phase':
            phase_noise = np.exp(1j * np.random.normal(0, rate, self.position_state.shape))
            self.position_state *= phase_noise
        elif model == 'amplitude':
            amplitude_noise = np.random.normal(1, rate, self.position_state.shape)
            self.position_state *= amplitude_noise

        # Normalize the state vector
        norm = np.linalg.norm(self.position_state, axis=0)
        self.position_state /= norm[:, np.newaxis]

    def shift(self, boundary='periodic'):
        new_state = np.zeros_like(self.position_state)
        if boundary == 'periodic':
            new_state[0, 1:] = self.position_state[0, :-1]
            new_state[0, 0] = self.position_state[0, -1]
            new_state[1, :-1] = self.position_state[1, 1:]
            new_state[1, -1] = self.position_state[1, 0]
        elif boundary == 'reflective':
            new_state[0, 1:] = self.position_state[0, :-1]
            new_state[1, :-1] = self.position_state[1, 1:]
            new_state[0, 0] = self.position_state[1, 0]
            new_state[1, -1] = self.position_state[0, -1]
        else:
            raise ValueError("Unsupported boundary condition")
        self.position_state = new_state

    """
    def shift(self):
        if self.topology == 'line' or self.topology == 'grid':
            # Shift operation for line or grid
            for d in range(self.dimension):
                self.position_states = np.roll(self.position_states, 1, axis=d+1)
        elif self.topology == 'network':
            # Shift operation on a network structure using adjacency matrix
            A = nx.adjacency_matrix(self.graph).toarray()
            self.position_states = A.dot(self.position_states)

    
    def step(self):
        self.apply_coin()
        self.shift()
        self.apply_decoherence()
    """

    def step(self, boundary='periodic'):
        self.apply_coin()
        self.apply_decoherence(rate=0.02)
        self.shift(boundary=boundary)

    def measure(self):
        probability_distribution = np.sum(np.abs(self.position_state)**2, axis=0)
        return probability_distribution
    
    def get_entanglement_measure(self):
        # Compute the density matrix
        density_matrix = np.tensordot(self.position_state, self.position_state.conj(), axes=0)

        # Trace out the position space to get reduced density matrix of the coin state
        reduced_density_matrix = np.trace(density_matrix, axis1=0, axis2=2)

        # Calculate the Von Neumann entropy as the entanglement measure
        eigenvalues = np.linalg.eigvalsh(reduced_density_matrix)
        # Filter out zero eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 0]
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return entropy
    
    def visualize_path_history(self):
        data = np.array([np.abs(state).sum(axis=0) for state in self.path_history])
        plt.imshow(data.T, interpolation='nearest', cmap='hot', aspect='auto')
        plt.colorbar()
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.title('Path History Heatmap')
        plt.show()
    
    def interactive_plot(state):
        fig = go.Figure(data=[go.Bar(x=list(range(len(state))), y=np.abs(state)**2)])
        fig.update_layout(title='Quantum State Probability Distribution',
                        xaxis_title='Position',
                        yaxis_title='Probability',
                        template='plotly_dark')
        fig.show()

    def update(frame, qw, line):
        qw.step()
        line.set_ydata(np.abs(qw.position_state[0])**2)
        return line,

    def animate_quantum_walk(qw):
        fig, ax = plt.subplots()
        line, = ax.plot(range(qw.num_positions), np.abs(qw.position_state[0])**2)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability')
        ax.set_title('Real-time Quantum Walk Animation')
        animation = FuncAnimation(fig, update, fargs=(qw, line), frames=200, interval=100, blit=True)
        plt.show()

    def on_coin_change(change):
        if change in ['Hadamard', 'Grover', 'Fourier']:
            qw.coin_type = change
            qw.coin_operation = qw.default_coin_operation()
            animate_quantum_walk(qw)
        else:
            print("Unsupported coin type")

    def adjust_parameters(change):
        if change['type'] == 'change' and change['name'] == 'value':
            qw.set_start_position(change['new'])
            animate_quantum_walk(qw)

    # more functions
    def set_graph(self, adjacency_matrix):
        if not isinstance(adjacency_matrix, np.ndarray) or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("Invalid adjacency matrix.")
        self.adjacency_matrix = adjacency_matrix
        self.num_positions = adjacency_matrix.shape[0]
        self.position_state = np.zeros((2, self.num_positions), dtype=complex)

    def graph_shift(self):
        new_state = np.zeros_like(self.position_state)
        for i in range(2):
            new_state[i] = self.adjacency_matrix.dot(self.position_state[i])
        self.position_state = new_state

    def temporal_coin_operation(self, step):
        if step % 5 == 0:
            self.coin_type = 'Grover'
        else:
            self.coin_type = 'Hadamard'
        self.apply_coin()

    def initialize_multiple_particles(self, positions):
        if len(positions) > self.num_positions:
            raise ValueError("More particles than positions available.")
        self.position_state = np.zeros((2, self.num_positions), dtype=complex)
        for pos in positions:
            self.position_state[0, pos] = 1 / np.sqrt(len(positions))
            self.position_state[1, pos] = 1 / np.sqrt(len(positions))

    def manage_interference(self):
        # Example: reduce interference by randomly applying phase shifts
        phases = np.exp(1j * np.pi * np.random.rand(self.num_positions))
        for i in range(2):
            self.position_state[i] *= phases

    def to_density_matrix(self):
        if not hasattr(self, 'density_matrix'):
            self.density_matrix = np.outer(self.position_state.flatten(), self.position_state.flatten().conj())

    def apply_decoherence_density_matrix(self, rate=0.01):
        decoherence_matrix = np.eye(self.num_positions) * rate
        self.density_matrix = (1 - rate) * self.density_matrix + decoherence_matrix * np.trace(self.density_matrix)

    def calculate_fidelity(self, target_state):
        current_state_vector = self.position_state.flatten()
        fidelity = np.abs(np.dot(current_optional_vector.conj(), target_state))**2
        return fidelity

    def update_boundary_conditions(self, boundary='open'):
        if boundary not in ['open', 'periodic', 'reflective']:
            raise ValueError("Unsupported boundary condition type.")
        self.boundary_type = boundary

    # oracle function
    def apply_oracle(self, oracle_function):
        """ Apply an oracle function to modify the state based on a decision problem. """
        for i in range(self.num_positions):
            if oracle_function(i):
                # Apply a phase flip if the oracle condition is met
                self.position_state[:, i] *= -1

    # other functions
    def apply_quantum_fourier_transform(self):
        """ Apply Quantum Fourier Transform to the position basis states. """
        from scipy.linalg import dft
        N = self.num_positions
        qft_matrix = dft(N, scale='sqrtn')  # Create a QFT matrix using SciPy
        self.position_state = qft_matrix.dot(self.position_state)

    def amplitude_amplification(self):
        """ Apply the quantum amplitude amplification, assuming a Grover iteration has been defined. """
        G = 2 * np.full((2, 2), 1/2) - np.eye(2)  # Grover diffusion operator
        self.position_state = np.tensordot(G, self.position_state, axes=[1, 0])
        self.apply_oracle(lambda x: self.position_state[0, x] > 0.1)  # Example condition
        self.position_state = np.tensordot(G, self.position_state, axes=[1, 0])

    def quantum_walk_search(self, target):
        """ Perform a quantum walk search for a target position. """
        steps = int(np.pi * np.sqrt(self.num_positions) / 4)  # Approximation of optimal steps
        for _ in range(steps):
            self.apply_coin()
            self.shift()
            # Mark the target with a phase flip
            self.position_state[:, target] *= -1
            self.apply_coin()  # Inversion about the average
            self.shift()
        return np.argmax(np.abs(self.position_state)**2)

    def interact_walkers(self, other_position_state, interaction_strength=0.1):
        """ Interact this walker's state with another walker's state. """
        if self.num_positions != other_position_state.shape[1]:
            raise ValueError("Both walkers must have the same number of positions.")
        # Simple interaction model: phase shift based on the other walker's state amplitude
        interaction_phase = np.exp(1j * interaction_strength * np.abs(other_position_state))
        self.position_state *= interaction_phase

    def prepare_quantum_state(self, angle_distribution):
        """ Prepare the quantum state with specific angles for superposition. """
        from numpy import cos, sin
        for i in range(self.num_positions):
            theta = angle_distribution[i]
            self.position_state[0, i] = cos(theta)
            self.position_state[1, i] = sin(theta)

    def entangle_positions(self, position1, position2):
        """ Entangle two positions using a controlled NOT gate after Hadamard. """
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        # Apply Hadamard to the first position
        self.position_state[:, position1] = np.dot(H, self.position_state[:, position1])
        # CNOT targeting the second position conditioned on the first
        self.position_state[1, position2] ^= self.position_state[1, position1]

    def continuous_time_step(self, adjacency_matrix, time_step=0.01):
        """ Evolve the quantum walk using the continuous-time model. """
        from scipy.linalg import expm
        # Hamiltonian for the continuous-time quantum walk
        H = -adjacency_matrix  # Negative of the adjacency matrix as a simple Hamiltonian
        U = expm(-1j * H * time_step)  # Time evolution operator
        self.position_state = U.dot(self.position_state)

    def apply_noise_channel(self, noise_type='depolarizing', noise_strength=0.01):
        """ Apply a quantum noise channel to the quantum state. """
        from numpy.random import rand
        if noise_type == 'depolarizing':
            for i in range(self.num_positions):
                if rand() < noise_strength:
                    # Randomize the state
                    self.position_state[:, i] = np.random.randn(2) + 1j * np.random.randn(2)
                    self.position_state[:, i] /= np.linalg.norm(self.position_state[:, i])

    def compress_quantum_state(self, compression_ratio=0.5):
        """ Compress the quantum state to reduce its size by a given ratio. """
        compressed_size = int(self.num_positions * compression_ratio)
        new_state = np.zeros((2, compressed_size), dtype=complex)
        step = self.num_positions // compressed_size
        for i in range(compressed_size):
            # Simple compression by averaging over 'step' positions
            new_state[:, i] = np.mean(self.position_state[:, i*step:(i+1)*step], axis=1)
        self.num_positions = compressed_reaching
        self.position_state = new_state
