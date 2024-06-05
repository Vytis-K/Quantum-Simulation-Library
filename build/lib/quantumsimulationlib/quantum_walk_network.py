import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider

class QuantumWalkOnNetwork:
    def __init__(self, num_nodes, graph_type='random', p=0.1, coin_type='Hadamard'):
        self.num_nodes = num_nodes
        self.coin_type = coin_type
        # Initialize graph
        if graph_type == 'random':
            self.graph = nx.gnp_random_graph(num_nodes, p)
        elif graph_type == 'small_world':
            self.graph = nx.watts_strogatz_graph(num_nodes, k=4, p=p)
        elif graph_type == 'scale_free':
            self.graph = nx.barabasi_albert_graph(num_nodes, m=2)
        else:
            raise ValueError("Unsupported graph type")

        # Create adjacency matrix
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()

        # Initialize position states in superposition
        self.position_states = np.zeros((2, num_nodes), dtype=complex)
        for node in range(num_nodes):
            self.position_states[0, node] = 1 / np.sqrt(num_nodes)
            self.position_states[1, node] = 1 / np.sqrt(num_nodes)

    def apply_coin(self):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        for node in range(self.num_nodes):
            self.position_states[:, node] = np.dot(H, self.position_states[:, node])

    def shift(self):
        new_state = np.zeros_like(self.position_states)
        for node in range(self.num_nodes):
            connected_nodes = self.adjacency_matrix[node]
            for connected_node, presence in enumerate(connected_nodes):
                if presence:
                    new_state[:, connected_node] += self.position_states[:, node]
        self.position_states = new_state / np.sqrt(np.sum(self.adjacency_matrix, axis=1, keepdims=True))

    def step(self):
        self.apply_coin()
        self.shift()

    def measure(self):
        probability_distribution = np.sum(np.abs(self.position_states)**2, axis=0)
        return probability_distribution

class IntegratedQuantumWalk:
    def __init__(self, num_positions, start_position=None, dimension=1, graph_type=None, coin_operation=None, coin_type='Hadamard'):
        self.dimension = dimension
        self.coin_type = coin_type
        self.graph_type = graph_type
        
        if graph_type:
            self.graph = self.create_graph(num_positions, graph_type)
            self.num_positions = len(self.graph.nodes())
            self.position_state = np.zeros((2, self.num_positions), dtype=complex)
        else:
            self.num_positions = num_positions
            self.position_state = np.zeros((2, self.num_positions), dtype=complex)
        
        if start_position is None:
            start_position = num_positions // 2 if not graph_type else 0
        self.position_state[0, start_position] = 1

        self.coin_operation = coin_operation if coin_operation else self.default_coin_operation(coin_type)

    def create_graph(self, num_positions, graph_type):
        if graph_type == 'random':
            return nx.gnp_random_graph(num_positions, p=0.1)
        elif graph_type == 'small_world':
            return nx.watts_strogatz_graph(num_positions, k=4, p=0.1)
        elif graph_type == 'scale_free':
            return nx.barabasi_albert_graph(num_positions, m=2)
        else:
            raise ValueError("Unsupported graph type")

    def default_coin_operation(self, coin_type):
        if coin_type == 'Hadamard':
            H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            return lambda pos_state: np.dot(H, pos_state)
        elif coin_type == 'Grover':
            G = 2 * np.full((2, 2), 1/2) - np.eye(2)
            return lambda pos_state: np.dot(G, pos_state)
        else:
            raise ValueError("Unsupported default coin type")

    def apply_coin(self):
        try:
            for pos in range(self.num_positions):
                self.position_state[:, pos] = self.coin_operation(self.position_state[:, pos])
        except Exception as e:
            raise RuntimeError(f"Error applying coin operation: {str(e)}")

    def apply_decoherence(self, rate=0.01):
        noise = (np.random.rand(*self.position_state.shape) < rate) * np.random.normal(loc=0.0, scale=1.0, size=self.position_state.shape)
        self.position_state += noise
        norm = np.sum(np.abs(self.position_state)**2)
        self.position_state /= np.sqrt(norm)

    def visualize_walk(self):
        node_intensities = np.abs(self.position_states[0])**2 + np.abs(self.position_states[1])**2
        node_colors = [plt.cm.viridis(i) for i in node_intensities]
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, node_color=node_colors, with_labels=True, node_size=300, edge_color='gray', alpha=0.7, cmap=plt.cm.viridis)
        plt.title('Current State of Quantum Walk on Network')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_intensities), vmax=max(node_intensities))), ax=plt.gca(), label='Probability Amplitude')
        plt.show()

    def update_graph_topology(self, new_graph_type, p=0.1):
        if new_graph_type == 'random':
            self.graph = nx.gnp_random_graph(self.num_nodes, p)
        elif new_graph_type == 'small_world':
            self.graph = nx.watts_strogatz_graph(self.num_nodes, k=4, p=p)
        elif new_graph_type == 'scale_free':
            self.graph = nx.barabasi_albert_graph(self.num_nodes, m=2)
        else:
            raise ValueError("Unsupported graph type")
        self.adjacency_matrix = nx.adjacency_matrix(self.graph).toarray()

    def calculate_entropy(self):
        probabilities = np.sum(np.abs(self.position_state)**2, axis=0)
        probabilities = probabilities[probabilities > 0]  # Avoid log(0) issues
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy

def interactive_walk_simulation(self):
    def update_walk(coin_type='Hadamard', decoherence_rate=0.01):
        self.coin_type = coin_type
        self.coin_operation = self.default_coin_operation(coin_type)
        self.apply_decoherence(decoherence_rate)
        self.step()
        self.visualize_walk()

    interact(update_walk, coin_type=['Hadamard', 'Grover', 'Fourier'], decoherence_rate=FloatSlider(min=0, max=0.1, step=0.01, value=0.01))

def animate_walk(self):
    fig, ax = plt.subplots()
    positions = np.arange(self.num_positions)
    line, = ax.plot(positions, np.zeros_like(positions), 'ro-')

    def update(frame):
        self.step()
        line.set_ydata(np.sum(np.abs(self.position_state)**2, axis=0))
        return line,

    ani = FuncAnimation(fig, update, frames=50, interval=100, blit=True)
    plt.show()

def shift(self, boundary='periodic'):
    if self.graph_type:
        # Network-based quantum walk shift
        new_state = np.zeros_like(self.position_state, dtype=complex)
        for node in self.graph.nodes():
            connected_nodes = list(self.graph.neighbors(node))
            for connected_node in connected_nodes:
                new_state[:, connected_node] += self.position_state[:, node] / len(connected_nodes)
        self.position_state = new_state
    else:
        # Traditional linear/grid quantum walk shift
        if boundary == 'periodic':
            # Apply periodic boundary conditions
            new_state = np.roll(self.position_state, shift=1, axis=1)
            new_state[:, 0] += np.roll(self.position_state, shift=-1, axis=1)[:, -1]
            self.position_state = new_state / 2
        elif boundary == 'reflective':
            # Apply reflective boundary conditions
            new_state = np.zeros_like(self.position_state, dtype=complex)
            new_state[:, 1:] += self.position_state[:, :-1]
            new_state[:, :-1] += self.position_state[:, 1:]
            new_state[:, 0] += self.position_state[:, 1]
            new_state[:, -1] += self.position_state[:, -2]
            self.position_state = new_state / 2
        else:
            raise ValueError("Unsupported boundary condition")


    def step(self, boundary='periodic'):
        self.apply_coin()
        self.apply_decoherence(rate=0.02)
        self.shift(boundary=boundary)

    def measure(self):
        probability_distribution = np.sum(np.abs(self.position_state)**2, axis=0)
        return probability_distribution
