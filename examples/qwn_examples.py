import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from ipywidgets import interact, FloatSlider

# Initialize a quantum walk on a scale-free network
num_nodes = 50  # Number of nodes in the graph
graph_type = 'scale_free'  # Type of graph
quantum_walk = QuantumWalkOnNetwork(num_nodes=num_nodes, graph_type=graph_type, coin_type='Hadamard')

# Run the quantum walk for a certain number of steps
num_steps = 100
for _ in range(num_steps):
    quantum_walk.step()

# Measure the probability distribution at the end of the walk
final_probabilities = quantum_walk.measure()

# Visualization of the final state using Plotly
fig = go.Figure(data=[go.Bar(x=list(range(num_nodes)), y=final_probabilities)])
fig.update_layout(title='Quantum Walk Final Probability Distribution on Scale-Free Network',
                  xaxis_title='Node',
                  yaxis_title='Probability',
                  template='plotly_dark')
fig.show()

# Additional visualization using Matplotlib for network layout
pos = nx.spring_layout(quantum_walk.graph)  # Position nodes using Fruchterman-Reingold force-directed algorithm
nx.draw(quantum_walk.graph, pos, node_size=50, node_color=final_probabilities, cmap=plt.cm.viridis)
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Probability')
plt.title('Node Probability Distribution in Network')
plt.show()
