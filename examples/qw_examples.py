import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider, IntSlider, Button

# Assuming the QuantumWalk class is defined in quantum_walk.py
from quantum_walk import QuantumWalk

# Example 1: Basic Quantum Walk Simulation and Visualization
num_positions = 10
start_position = 5

# Initialize the quantum walk
qw = QuantumWalk(num_positions, start_position)

# Perform a number of steps in the quantum walk
num_steps = 20
for _ in range(num_steps):
    qw.step()

# Measure the probabilities
probabilities = qw.measure()
print("Probabilities after the walk:")
print(probabilities)

# Visualize the path history
qw.visualize_path_history()

# Example 2: Simulate Decoherence and Measure Entanglement
qw.reset()
decoherence_rate = 0.05

for _ in range(num_steps):
    qw.step()
    qw.apply_decoherence(rate=decoherence_rate)

# Measure and print entanglement
entanglement_measure = qw.get_entanglement_measure()
print("Entanglement measure after decoherence:")
print(entanglement_measure)

# Example 3: Interactive Quantum Walk with Plotly
def interactive_plot(state):
    fig = go.Figure(data=[go.Bar(x=list(range(len(state))), y=np.abs(state)**2)])
    fig.update_layout(title='Quantum State Probability Distribution',
                      xaxis_title='Position',
                      yaxis_title='Probability',
                      template='plotly_dark')
    fig.show()

interactive_plot(qw.position_state[0])

# Example 4: Animate Quantum Walk
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

animate_quantum_walk(qw)

# Example 5: Adaptive Coin Operation and Graph-Based Quantum Walk
qw.reset()

# Create a random adjacency matrix for a graph-based walk
adjacency_matrix = np.random.randint(0, 2, size=(num_positions, num_positions))
adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T  # Make it symmetric

qw.set_graph(adjacency_matrix)

# Perform a graph-based walk with temporal coin operation
for step in range(num_steps):
    qw.temporal_coin_operation(step)
    qw.graph_shift()

# Visualize the final state
qw.visualize_path_history()

# Example 6: Quantum Walk with Noise Channel and Fidelity Calculation
target_state = np.ones((2, num_positions), dtype=complex) / np.sqrt(num_positions)

qw.reset()
noise_strength = 0.1
for _ in range(num_steps):
    qw.apply_noise_channel(noise_type='depolarizing', noise_strength=noise_strength)
    qw.step()

fidelity = qw.calculate_fidelity(target_state)
print("Fidelity with target state after noise:")
print(fidelity)

# Example 7: Interactive Simulation with Widgets
def update_walk(steps, decoherence_rate):
    qw.reset()
    for _ in range(steps):
        qw.step()
        qw.apply_decoherence(rate=decoherence_rate)
    qw.visualize_path_history()

interact(update_walk, steps=IntSlider(min=1, max=50, step=1, value=10),
         decoherence_rate=FloatSlider(min=0, max=0.1, step=0.01, value=0.01))

# Example 8: Continuous-Time Quantum Walk Simulation
qw.reset()

# Create an adjacency matrix for the continuous-time walk
adjacency_matrix = np.random.randint(0, 2, size=(num_positions, num_positions))
adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T

qw.set_graph(adjacency_matrix)
time_step = 0.1

for _ in range(num_steps):
    qw.continuous_time_quantum_walk(time_step=time_step)

# Measure and print the final state probabilities
probabilities = qw.measure()
print("Probabilities after continuous-time quantum walk:")
print(probabilities)
