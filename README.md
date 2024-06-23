# Quantum Simulation Package

This is a  quantum simulation package designed to facilitate quantum computing research and applications. The package includes classes and methods for simulating quantum walks, visualizations, quantum machine learning, and more. It supports various coin operations, network topologies, and advanced features such as entanglement and decoherence.

## Features

- Quantum Walks with Hadamard, Grover, and Fourier coins
- Visualization tools including real-time animations and heatmaps
- Basic Graphical User Interface (GUI) for interactive simulations
- Quantum machine learning and entangled quantum walks
- Multidimensional and higher-dimensional quantum walks
- Integration with Qiskit for advanced quantum computing applications

## Classes and Methods

### 1. `QuantumWalk`

This class simulates a quantum walk on a linear or grid topology with support for various coin operations and decoherence models.

#### Methods:
- `__init__(self, num_positions, start_position, coin_operation=None)`: Initializes the quantum walk with the given number of positions and start position.
- `reset(self)`: Resets the quantum walk to the initial state.
- `set_start_position(self, position)`: Sets the start position for the quantum walk.
- `create_matrix_operation(self, matrix)`: Creates a custom coin operation from a given matrix.
- `default_coin_operation(self)`: Returns the default coin operation based on the coin type.
- `apply_coin(self)`: Applies the coin operation to the position states.
- `apply_decoherence(self, rate=0.01, model='gaussian')`: Applies decoherence to the position states using different models (Gaussian, phase, amplitude).
- `shift(self, boundary='periodic')`: Shifts the position states with specified boundary conditions (periodic, reflective).
- `step(self, boundary='periodic')`: Performs a single step of the quantum walk.
- `measure(self)`: Measures the probability distribution of the position states.
- `get_entanglement_measure(self)`: Calculates the entanglement measure (Von Neumann entropy) of the position states.
- `visualize_path_history(self)`: Visualizes the path history of the quantum walk as a heatmap.
- `interactive_plot(state)`: Creates an interactive plot of the quantum state probability distribution using Plotly.
- `animate_quantum_walk(qw)`: Animates the quantum walk in real-time using Matplotlib.
- `apply_oracle(self, oracle_function)`: Applies an oracle function to the position states for quantum search algorithms.
- `apply_quantum_fourier_transform(self)`: Applies the Quantum Fourier Transform to the position states.
- `amplitude_amplification(self)`: Performs amplitude amplification for quantum search algorithms.
- `quantum_walk_search(self, target)`: Performs a quantum walk search for a target position.
- `interact_walkers(self, other_position_state, interaction_strength=0.1)`: Interacts the current walker’s state with another walker’s state.
- `prepare_quantum_state(self, angle_distribution)`: Prepares the quantum state with specific angles for superposition.
- `entangle_positions(self, position1, position2)`: Entangles two positions using a controlled NOT gate after Hadamard operation.
- `continuous_time_step(self, adjacency_matrix, time_step=0.01)`: Evolve the quantum walk using the continuous-time model.
- `apply_noise_channel(self, noise_type='depolarizing', noise_strength=0.01)`: Applies a quantum noise channel to the quantum state.
- `compress_quantum_state(self, compression_ratio=0.5)`: Compresses the quantum state to reduce its size by a given ratio.
- `simulate_phase_kickback(self, control_qubit_position, target_qubit_position)`: Simulates phase kickback effect between control and target qubits.
- `compress_state(self, factor)`: Compresses the quantum state by a given factor to reduce its size.
- `update_topology(self, new_adjacency_matrix)`: Dynamically updates the graph topology during the quantum walk.
- `interact_with_environment(self, interaction_strength)`: Introduces environmental interaction during the quantum walk.
- `analyze_entanglement(self)`: Analyzes and returns the degree of entanglement across the quantum state.
- `dynamic_parameter_tuning(self)`: Dynamically tunes parameters of the quantum walk based on real-time measurements.
- `apply_quantum_error_correction(self)`: Applies a basic quantum error correction code to each position state.
- `initialize_gaussian_state(self, mean, variance)`: Initializes the position states with a Gaussian distribution.
- `interactive_quantum_walk(self)`: Provides an interactive simulation of the quantum walk process.
- `continuous_time_quantum_walk(self, time_step=0.1)`: Simulates a continuous-time quantum walk using the adjacency matrix.

### 2. `QuantumWalkOnNetwork`

This class extends the `QuantumWalk` class to support quantum walks on various network topologies.

#### Methods:
- `__init__(self, num_nodes, graph_type='random', p=0.1, coin_type='Hadamard')`: Initializes the quantum walk on a network with the given number of nodes and graph type.
- `simulate_entanglement_dynamics(self)`: Simulates the development of entanglement across the network.
- `actively_disentangle_nodes(self)`: Actively disentangles nodes based on specific conditions or metrics.
- `adaptive_quantum_walk(self, optimization_goal)`: Adjusts the quantum walk dynamically to optimize a given goal.
- `dynamic_network_rewiring(self)`: Rewires the network connections based on the quantum state to optimize performance.
- `quantum_teleportation(self, sender, receiver, entangled_pair)`: Simulates quantum teleportation between two nodes in the network using an entangled pair.
- `create_bell_pair(self, nodes)`: Initializes a Bell pair between two nodes.
- `perform_bell_measurement(self, psi, phi)`: Simulates a Bell measurement and returns the result as a string of bits.
- `dynamic_quantum_routing(self)`: Dynamically routes quantum information in the network to optimize path fidelity.
- `simulate_quantum_transmission(self, path)`: Simulates the transmission of a quantum state along a path and returns the fidelity.
- `entanglement_percolation(self)`: Studies entanglement percolation in the quantum network.
- `calculate_quantum_centrality(self)`: Calculates node centrality based on the steady-state distribution of the quantum walk.
- `detect_communities(self)`: Detects communities in the graph using the pattern of quantum coherence.
- `simulate_state_diffusion(self, start_node)`: Simulates the diffusion of quantum information from a specific start node.
- `visualize_heatmap_evolution(self)`: Visualizes the evolution of the quantum walk as a heatmap.
- `dynamic_node_interaction(self)`: Adjusts interactions dynamically based on the state of the quantum walk.

### 3. `EntangledQuantumWalk`

This class extends the `QuantumWalk` class to support quantum walks with multiple particles and entanglement.

#### Methods:
- `__init__(self, num_positions, num_particles, dimension=1, topology='line', coin_type='Hadamard')`: Initializes the entangled quantum walk with the given number of positions and particles.
- `generate_entanglement(self, particles)`: Generates entanglement between specified particles.
- `apply_multi_coin(self)`: Applies different coin operations based on the state configuration.
- `update_topology(self, new_topology, connections=None)`: Updates the topology of the quantum walk.
- `measure_in_basis(self, basis='computational')`: Measures the quantum state in the specified basis.
- `visualize_entanglement(self)`: Visualizes the pairwise entanglement between particles.
- `perform_state_tomography(self)`: Performs state tomography to reconstruct the density matrix.
- `adapt_coin_operation(self, condition)`: Adapts the coin operation based on specified conditions.
- `integrate_memory_effects(self, memory_strength=0.1)`: Integrates memory effects into the quantum walk.
- `simulate_particle_interactions(self, interaction_strength=0.05)`: Simulates interactions between particles.
- `apply_time_dependent_dynamics(self, time_step)`: Applies time-dependent dynamics to the quantum walk.
- `propagate_entanglement(self)`: Propagates entanglement through the system.
- `adjust_topology_dynamically(self, adjustment_criteria)`: Adjusts the topology dynamically based on specified criteria.
- `control_entanglement_temporally(self, control_function, time_steps)`: Controls entanglement temporally based on a control function.
- `manage_quantum_interference(self, interference_strategy)`: Manages quantum interference effects.
- `measurement_driven_walk(self)`: Adjusts the quantum walk based on real-time measurement outcomes.
- `entanglement_filtering(self, filter_function)`: Applies a filter to selectively adjust entanglement.
- `dynamic_entanglement_generation(self, control_sequence)`: Dynamically generates entanglement based on a sequence of control operations.
- `simulate_decoherence(self, decoherence_rate)`: Simulates the effect of decoherence on the entangled quantum states.
- `entanglement_based_measurement(self)`: Measures the quantum state using an entanglement-based protocol.
- `quantum_decision_making(self, utility_function, decision_threshold=0.6, feedback_iterations=5)`: Utilizes the entangled quantum walk to make decisions based on probability distributions modified by a utility function.
- `simulate_noise_effects(self, noise_types)`: Simulates various types of noise on the quantum walk.

## Installation

To install the package, use pip:
```bash
pip install quantum-simulation-package
```

##

 Usage

Refer to the examples provided in the documentation for detailed usage of the classes and methods. You can also find interactive notebooks and scripts in the examples directory of the repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributions

Contributions are welcome! Please submit issues or pull requests to the GitHub repository.

## Contact

For questions or feedback, please contact at vytis000@gmail.com.

---

This detailed README provides a comprehensive overview of the package, including its features, classes, and methods, along with installation instructions and usage examples.