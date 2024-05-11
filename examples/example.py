import quantumsimulationlib


# multi-dimensional quantum walk

# Example Usage
dimensions = 2  # For a 2D quantum walk
size = 10       # 10x10 grid
start_position = (size // 2, size // 2)  # Start at the center of the grid
quantum_walk = MultiDimensionalQuantumWalk(dimensions, size, start_position)

# Perform some steps
steps = 5
for _ in range(steps):
    quantum_walk.step()

# Measure the probability distribution
print("Probability distribution after {} steps:".format(steps))
print(quantum_walk.measure())

# quantum walk network

# Example Usage
num_nodes = 10  # Number of nodes in the graph
quantum_walk = QuantumWalkOnNetwork(num_nodes, graph_type='small_world', p=0.2)

# Perform some steps
steps = 5
for _ in range(steps):
    quantum_walk.step()

# Measure the probability distribution
print("Probability distribution after {} steps:".format(steps))
print(quantum_walk.measure())

# quantum walk network

# Example for a small world network-based quantum walk
qw_network = IntegratedQuantumWalk(num_positions=10, graph_type='small_world')
qw_network.step()
print("Network-based Quantum Walk Probability Distribution:", qw_network.measure())

# quantum machine learning

interact(update_simulation, parameters="0.5,0.5,0.5")