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
        if self.coin_type == 'Hadamard':
            H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        elif self.coin_type == 'Grover':
            G = 2 * np.full((2, 2), 0.5) - np.eye(2)
        elif self.coin_type == 'Fourier':
            F = np.array([[1, 1], [1, -1j]]) / np.sqrt(2)
        else:
            raise ValueError("Unsupported coin type")
        
        self.position_states = np.apply_along_axis(lambda x: np.dot(locals()[self.coin_type], x), 0, self.position_states)

    def apply_decoherence(self, rate=0.01):
        noise = (np.random.rand(*self.position_state.shape) < rate) * np.random.normal(loc=0.0, scale=1.0, size=self.position_state.shape)
        self.position_state += noise
        norm = np.sum(np.abs(self.position_state)**2)
        self.position_state /= np.sqrt(norm)

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
    self.density_matrix = np.tensordot(self.position_state, np.conj(self.position_state), axes=0)

def apply_decoherence_density_matrix(self, rate=0.01):
    decoherence_matrix = np.eye(self.num_positions) * rate
    self.density_matrix = (1 - rate) * self.density_matrix + decoherence_matrix * np.trace(self.density_matrix)

def calculate_fidelity(self, target_state):
    current_state_vector = self.position_state.flatten()
    fidelity = np.abs(np.dot(current_optional_vector.conj(), target_state))**2
    return fidelity

def update_boundary_conditions(self, boundary='open'):
    self.boundary_type = boundary
