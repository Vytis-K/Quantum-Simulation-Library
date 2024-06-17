import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go

class QuantumWalkTester:
    def __init__(self, num_positions, start_position, coin_type='Hadamard'):
        self.qw = QuantumWalk(num_positions, start_position, coin_type)
        self.num_positions = num_positions
        self.start_position = start_position

    def run_steps(self, num_steps=100, boundary='periodic'):
        self.qw.reset()
        self.qw.set_start_position(self.start_position)
        for _ in range(num_steps):
            self.qw.step(boundary=boundary)
        return self.qw.measure()

    def visualize_path_history(self):
        self.qw.visualize_path_history()

    def animate_quantum_walk(self, num_frames=200):
        def update(frame, qw, line):
            qw.step()
            line.set_ydata(np.abs(qw.position_state[0])**2)
            return line,

        fig, ax = plt.subplots()
        line, = ax.plot(range(self.qw.num_positions), np.abs(self.qw.position_state[0])**2)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability')
        ax.set_title('Real-time Quantum Walk Animation')
        animation = FuncAnimation(fig, update, fargs=(self.qw, line), frames=num_frames, interval=100, blit=True)
        plt.show()

    def test_coin_operations(self):
        self.qw.apply_coin()
        probabilities = np.abs(self.qw.position_state)**2
        print(f"Probabilities after applying {self.qw.coin_type} coin: {probabilities}")

    def test_decoherence(self, rate=0.01, model='gaussian'):
        self.qw.apply_decoherence(rate, model)
        probabilities = np.abs(self.qw.position_state)**2
        print(f"Probabilities after applying {model} decoherence: {probabilities}")

    def test_entanglement(self):
        entanglement = self.qw.get_entanglement_measure()
        print(f"Entanglement measure: {entanglement}")

    def interactive_plot(self):
        state = self.qw.position_state.flatten()
        fig = go.Figure(data=[go.Bar(x=list(range(len(state))), y=np.abs(state)**2)])
        fig.update_layout(title='Quantum State Probability Distribution',
                          xaxis_title='Position',
                          yaxis_title='Probability',
                          template='plotly_dark')
        fig.show()

    def set_coin_type(self, coin_type):
        self.qw.coin_type = coin_type
        self.qw.coin_operation = self.qw.default_coin_operation()

    def reset(self):
        self.qw.reset()

    def setUp(self):
        self.num_positions = 5
        self.start_position = 2
        self.qw = QuantumWalk(self.num_positions, self.start_position)
    
    def test_initial_state(self):
        expected_state = np.zeros((2, self.num_positions), dtype=complex)
        expected_state[0, self.start_position] = 1
        np.testing.assert_array_equal(self.qw.position_state, expected_state)

    def test_apply_coin(self):
        self.qw.apply_coin()
        state_after_coin = np.dot(np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.array([1, 0], dtype=complex))
        expected_state = np.zeros((2, self.num_positions), dtype=complex)
        expected_state[:, self.start_position] = state_after_coin
        np.testing.assert_almost_equal(self.qw.position_state, expected_state)

    def test_shift(self):
        self.qw.shift()
        shifted_state = np.zeros((2, self.num_positions), dtype=complex)
        shifted_state[0, (self.start_position + 1) % self.num_positions] = 1
        np.testing.assert_array_equal(self.qw.position_state, shifted_state)

    def test_measure(self):
        probabilities = self.qw.measure()
        expected_probabilities = np.zeros(self.num_positions)
        expected_probabilities[self.start_position] = 1
        np.testing.assert_array_equal(probabilities, expected_probabilities)
    
    def test_step(self):
        initial_prob = self.qw.measure()[self.start_position]
        self.qw.step()
        final_prob = self.qw.measure()[self.start_position]
        self.assertNotEqual(initial_prob, final_prob)

    def test_entanglement_measure(self):
        entanglement = self.qw.get_entanglement_measure()
        self.assertTrue(entanglement >= 0)

    def test_reset(self):
        self.qw.step()
        self.qw.reset()
        expected_state = np.zeros((2, self.num_positions), dtype=complex)
        expected_state[0, self.start_position] = 1
        np.testing.assert_array_equal(self.qw.position_state, expected_state)

    def test_set_start_position(self):
        new_start = 3
        self.qw.set_start_position(new_start)
        expected_state = np.zeros((2, self.num_positions), dtype=complex)
        expected_state[0, new_start] = 1
        np.testing.assert_array_equal(self.qw.position_state, expected_state)

    def test_apply_decoherence(self):
        self.qw.apply_decoherence(rate=0.05)
        norm = np.linalg.norm(self.qw.position_state)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_boundary_conditions(self):
        self.qw.shift(boundary='periodic')
        self.assertTrue(np.allclose(np.sum(np.abs(self.qw.position_state)**2), 1))

        self.qw.shift(boundary='reflective')
        self.assertTrue(np.allclose(np.sum(np.abs(self.qw.position_state)**2), 1))

    def test_graph_shift(self):
        adjacency_matrix = np.array([[0, 1, 0, 0, 0],
                                     [1, 0, 1, 0, 0],
                                     [0, 1, 0, 1, 0],
                                     [0, 0, 1, 0, 1],
                                     [0, 0, 0, 1, 0]])
        self.qw.set_graph(adjacency_matrix)
        self.qw.graph_shift()
        self.assertTrue(np.allclose(np.sum(np.abs(self.qw.position_state)**2), 1))

    def test_temporal_coin_operation(self):
        for step in range(10):
            self.qw.temporal_coin_operation(step)
            self.assertIn(self.qw.coin_type, ['Hadamard', 'Grover'])

    def test_initialize_multiple_particles(self):
        positions = [0, 2, 4]
        self.qw.initialize_multiple_particles(positions)
        total_probability = np.sum(np.abs(self.qw.position_state)**2)
        self.assertAlmostEqual(total_probability, 1.0, places=5)

    def test_manage_interference(self):
        initial_state = self.qw.position_state.copy()
        self.qw.manage_interference()
        final_state = self.qw.position_state
        self.assertFalse(np.allclose(initial_state, final_state))

    def test_density_matrix(self):
        self.qw.to_density_matrix()
        self.assertEqual(self.qw.density_matrix.shape, (2*self.num_positions, 2*self.num_positions))

    def test_apply_decoherence_density_matrix(self):
        self.qw.to_density_matrix()
        self.qw.apply_decoherence_density_matrix(rate=0.05)
        trace = np.trace(self.qw.density_matrix)
        self.assertAlmostEqual(trace, 1.0, places=5)

    def test_fidelity(self):
        target_state = np.zeros((2, self.num_positions), dtype=complex)
        target_state[0, self.start_position] = 1
        fidelity = self.qw.calculate_fidelity(target_state)
        self.assertTrue(0 <= fidelity <= 1)

# Example usage
tester = QuantumWalkTester(num_positions=50, start_position=25, coin_type='Hadamard')
tester.run_steps(num_steps=100)
tester.visualize_path_history()
tester.animate_quantum_walk(num_frames=200)
tester.test_coin_operations()
tester.test_decoherence(rate=0.02, model='phase')
tester.test_entanglement()
tester.set_coin_type('Grover')
tester.run_steps(num_steps=50)
tester.visualize_path_history()
tester.interactive_plot()
tester.reset()
