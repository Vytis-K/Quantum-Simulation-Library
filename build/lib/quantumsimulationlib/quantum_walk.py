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
            self.coin_operation = self.default_coin_operation  # Default operation
        elif isinstance(coin_operation, np.ndarray):
            self.coin_operation = self.create_matrix_operation(coin_operation)
        elif callable(coin_operation):
            self.coin_operation = coin_operation
        else:
            raise TypeError("coin_operation must be either a numpy array or a callable.")

    def create_matrix_operation(self, matrix):
        if matrix.shape != (2, 2):
            raise ValueError("Coin matrix must be 2x2.")
        return lambda pos_state: np.dot(matrix, pos_state)

    def default_coin_operation(self):
        # Implement default coin operations like Hadamard, Grover, etc.
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
        try:
            self.position_state = self.coin_operation(self.position_state)
        except Exception as e:
            raise RuntimeError(f"Error applying coin operation: {str(e)}")

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

    def step(self, boundary='periodic'):
        self.apply_coin()
        self.apply_decoherence(rate=0.02)
        self.shift(boundary=boundary)

    def measure(self):
        probability_distribution = np.sum(np.abs(self.position_state)**2, axis=0)
        return probability_distribution
    
def interactive_plot(state):
    fig = go.Figure(data=[go.Bar(x=list(range(len(state))), y=np.abs(state)**2)])
    fig.update_layout(title='Quantum State Probability Distribution',
                      xaxis_title='Position',
                      yaxis_title='Probability',
                      template='plotly_dark')
    fig.show()

def update(frame, qw, line):
    qw.step()
    line.set_ydata(np.abs(qw.position_state[0])**2)  # Update the data.
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
    # Example of changing the coin operation
    if change == 'Hadamard':
        qw.coin_operation = qw.default_coin_operation('Hadamard')
    elif change == 'Grover':
        qw.coin_operation = qw.default_coin_operation('Grover')
    animate_quantum_walk(qw)
