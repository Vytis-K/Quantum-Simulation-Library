import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_hermite
from scipy.linalg import expm, logm
from mpl_toolkits.mplot3d import Axes3D

class SuperconductingQubit:
    def __init__(self, EJ, EC, phi_ext):
        self.EJ = EJ  # Josephson energy
        self.EC = EC  # Charging energy
        self.phi_ext = phi_ext  # External magnetic flux
    
    def hamiltonian(self, phi, q):
        """ Hamiltonian for a superconducting qubit """
        return 4 * self.EC * (q ** 2) - self.EJ * np.cos(phi - self.phi_ext)
    
    def equations_of_motion(self, t, y):
        """ Equations of motion for the superconducting qubit """
        phi, q = y
        dphi_dt = 8 * np.pi * self.EC * q
        dq_dt = -self.EJ * np.sin(phi - self.phi_ext)
        return [dphi_dt, dq_dt]
    
    def simulate(self, initial_conditions, t_span):
        """ Simulate the dynamics of the qubit """
        sol = solve_ivp(self.equations_of_motion, t_span, initial_conditions, method='RK45')
        return sol.t, sol.y

    def plot_dynamics(self, t, phi, q):
        """ Plot the dynamics of the qubit """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(t, phi, label='Phase (phi)')
        plt.xlabel('Time')
        plt.ylabel('Phase')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(t, q, label='Charge (q)', color='r')
        plt.xlabel('Time')
        plt.ylabel('Charge')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def calculate_energy(self, phi, q):
        """ Calculate the total energy of the qubit """
        return self.hamiltonian(phi, q)
    
    def plot_phase_space(self, t, phi, q):
        """ Plot the phase space of the qubit """
        plt.figure(figsize=(8, 6))
        plt.plot(phi, q)
        plt.xlabel('Phase (phi)')
        plt.ylabel('Charge (q)')
        plt.title('Phase Space Trajectory')
        plt.grid(True)
        plt.show()
    
    def eigenenergies(self, num_levels=10):
        """ Calculate the eigenenergies of the qubit """
        # Discretize phase
        N = 100  # Number of points in phase space
        phi_max = 2 * np.pi
        phi_vals = np.linspace(-phi_max, phi_max, N)
        
        # Kinetic energy (charge) term
        q_vals = np.arange(-N//2, N//2)
        T = 4 * self.EC * (q_vals ** 2)
        
        # Potential energy term
        V = -self.EJ * np.cos(phi_vals - self.phi_ext)
        
        # Diagonalize Hamiltonian
        energies, _ = eigh_tridiagonal(V, np.full(N-1, self.EJ/2))
        
        return energies[:num_levels]

    def plot_eigenenergies(self, num_levels=10):
        """ Plot the eigenenergies of the qubit """
        energies = self.eigenenergies(num_levels)
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, num_levels + 1), energies, 'o-')
        plt.xlabel('Energy Level')
        plt.ylabel('Energy')
        plt.title('Eigenenergies of the Superconducting Qubit')
        plt.grid(True)
        plt.show()
    
    def wigner_function(self, alpha_re, alpha_im):
        """ Calculate the Wigner function of the qubit """
        alpha = alpha_re + 1j * alpha_im
        beta = np.conj(alpha)
        w = 2 * np.exp(-2 * np.abs(alpha) ** 2) * np.sum(
            [(-1) ** n / np.math.factorial(n) * (2 * np.abs(alpha) ** 2) ** n for n in range(50)]
        )
        return w.real
    
    def plot_wigner_function(self, re_range, im_range):
        """ Plot the Wigner function of the qubit """
        re_vals = np.linspace(-re_range, re_range, 100)
        im_vals = np.linspace(-im_range, im_range, 100)
        W = np.array([[self.wigner_function(re, im) for im in im_vals] for re in re_vals])
        
        plt.figure(figsize=(8, 6))
        plt.contourf(re_vals, im_vals, W, levels=100, cmap='RdBu')
        plt.colorbar(label='Wigner Function')
        plt.xlabel('Re(α)')
        plt.ylabel('Im(α)')
        plt.title('Wigner Function')
        plt.show()
    
    def entanglement_entropy(self, rho):
        """ Calculate the entanglement entropy of the qubit """
        eigenvalues = np.linalg.eigvalsh(rho)
        entropy = -np.sum([p * np.log2(p) for p in eigenvalues if p > 0])
        return entropy
    
    def fidelity(self, rho1, rho2):
        """ Calculate the fidelity between two quantum states """
        sqrt_rho1 = expm(0.5 * np.logm(rho1))
        product = sqrt_rho1 @ rho2 @ sqrt_rho1
        return np.trace(np.sqrt(product)).real
    
    def state_tomography(self, measurements):
        """ Perform state tomography to reconstruct the density matrix """
        # For simplicity, assume measurements are given as a dictionary of expectation values
        # for Pauli matrices {I, X, Y, Z}
        rho = 0.5 * (np.eye(2) + 
                     measurements['X'] * np.array([[0, 1], [1, 0]]) +
                     measurements['Y'] * np.array([[0, -1j], [1j, 0]]) +
                     measurements['Z'] * np.array([[1, 0], [0, -1]]))
        return rho

    def evolve_state(self, psi0, t_span):
        """ Evolve the given quantum state psi0 using the Hamiltonian """
        def schrodinger(t, psi):
            return -1j * self.hamiltonian_matrix() @ psi
        
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval)
        return sol.t, sol.y

    def hamiltonian_matrix(self):
        """ Hamiltonian matrix for the qubit """
        N = 2
        H = np.zeros((N, N), dtype=complex)
        for n in range(N):
            H[n, n] = 4 * self.EC * (n ** 2)
            if n < N - 1:
                H[n, n+1] = -self.EJ / 2
                H[n+1, n] = -self.EJ / 2
        return H

    def apply_gate(self, psi, gate):
        """ Apply a quantum gate to the qubit state """
        gates = {
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]]),
            'H': (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        }
        return gates[gate] @ psi

    def plot_bloch_sphere(self, psi):
        """ Plot the state of the qubit on the Bloch sphere """
        def bloch_coordinates(psi):
            a, b = psi
            x = 2 * (a.real * b.real + a.imag * b.imag)
            y = 2 * (a.imag * b.real - a.real * b.imag)
            z = np.abs(a)**2 - np.abs(b)**2
            return x, y, z
        
        x, y, z = bloch_coordinates(psi)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, x, y, z, color='r', arrow_length_ratio=0.1)
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        X = np.outer(np.cos(u), np.sin(v))
        Y = np.outer(np.sin(u), np.sin(v))
        Z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(X, Y, Z, color='b', alpha=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def simulate_decoherence(self, rho, t_span, gamma1, gamma2):
        """ Simulate the decoherence of the qubit """
        def lindblad(t, rho):
            rho = rho.reshape((2, 2))
            L1 = np.sqrt(gamma1) * np.array([[0, 1], [0, 0]])
            L2 = np.sqrt(gamma2) * np.array([[0, 0], [1, 0]])
            Ld = L1 @ rho @ L1.conj().T + L2 @ rho @ L2.conj().T
            L = -0.5 * (L1.conj().T @ L1 + L2.conj().T @ L2) @ rho - rho @ 0.5 * (L1.conj().T @ L1 + L2.conj().T @ L2)
            return (L + Ld).reshape(-1)
        
        rho0 = rho.reshape(-1)
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        sol = solve_ivp(lindblad, t_span, rho0, t_eval=t_eval)
        return sol.t, sol.y.reshape((100, 2, 2))

    def fourier_transform_dynamics(self, t, signal):
        """ Compute the Fourier transform of the qubit dynamics """
        signal_ft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), t[1] - t[0])
        return freqs, np.abs(signal_ft)

    def plot_fourier_transform(self, t, phi, q):
        """ Plot the Fourier transform of the phase and charge dynamics """
        freqs_phi, phi_ft = self.fourier_transform_dynamics(t, phi)
        freqs_q, q_ft = self.fourier_transform_dynamics(t, q)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(freqs_phi, phi_ft, label='Phase (phi)')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform of Phase Dynamics')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(freqs_q, q_ft, label='Charge (q)', color='r')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Fourier Transform of Charge Dynamics')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def measure_state(self, psi):
        """ Simulate the measurement of the qubit state """
        probabilities = np.abs(psi)**2
        outcome = np.random.choice(len(psi), p=probabilities)
        measured_state = np.zeros_like(psi)
        measured_state[outcome] = 1
        return measured_state, outcome

    def simulate_measurements(self, psi, num_measurements):
        """ Simulate multiple measurements of the qubit state """
        outcomes = []
        for _ in range(num_measurements):
            _, outcome = self.measure_state(psi)
            outcomes.append(outcome)
        return outcomes

    def expectation_value(self, psi, operator):
        """ Calculate the expectation value of an operator for the qubit state """
        return np.vdot(psi, operator @ psi).real

    def calculate_expectations(self, psi):
        """ Calculate expectation values of Pauli matrices for the qubit state """
        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        
        exp_x = self.expectation_value(psi, pauli_x)
        exp_y = self.expectation_value(psi, pauli_y)
        exp_z = self.expectation_value(psi, pauli_z)
        
        return {'X': exp_x, 'Y': exp_y, 'Z': exp_z}

    def time_dependent_hamiltonian(self, t, args):
        """ Example time-dependent Hamiltonian """
        omega = args.get('omega', 1.0)
        return self.hamiltonian_matrix() * np.cos(omega * t)

    def simulate_time_dependent(self, psi0, t_span, args):
        """ Simulate the qubit dynamics with a time-dependent Hamiltonian """
        def schrodinger(t, psi):
            H = self.time_dependent_hamiltonian(t, args)
            return -1j * H @ psi
        
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval)
        return sol.t, sol.y

    def calculate_purity(self, rho):
        """ Calculate the purity of the quantum state """
        return np.trace(rho @ rho).real

    def fidelity_over_time(self, psi0, t_span, reference_state, args=None):
        """ Compute the fidelity of the quantum state with respect to a reference state over time """
        if args is None:
            args = {}

        def schrodinger(t, psi):
            H = self.hamiltonian_matrix() if 'time_dependent' not in args else self.time_dependent_hamiltonian(t, args)
            return -1j * H @ psi
        
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval)
        
        fidelities = [self.fidelity(sol.y[:, i], reference_state) for i in range(len(sol.t))]
        return sol.t, fidelities

    def plot_fidelity_over_time(self, t, fidelities):
        """ Plot the fidelity of the quantum state with respect to a reference state over time """
        plt.figure(figsize=(8, 6))
        plt.plot(t, fidelities, label='Fidelity')
        plt.xlabel('Time')
        plt.ylabel('Fidelity')
        plt.title('Fidelity Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def simulate_rabi_oscillations(self, psi0, t_span, omega):
        """ Simulate Rabi oscillations of the qubit under a driving field """
        def rabi_hamiltonian(t):
            H0 = self.hamiltonian_matrix()
            H1 = omega * np.array([[0, 1], [1, 0]])
            return H0 + H1
        
        def schrodinger(t, psi):
            return -1j * rabi_hamiltonian(t) @ psi
        
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval)
        return sol.t, sol.y

    def plot_rabi_oscillations(self, t, psi):
        """ Plot Rabi oscillations of the qubit """
        probabilities = np.abs(psi)**2
        plt.figure(figsize=(8, 6))
        plt.plot(t, probabilities[0, :], label='Ground State')
        plt.plot(t, probabilities[1, :], label='Excited State')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title('Rabi Oscillations')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_berry_phase(self, psi0, t_span):
        """ Calculate the Berry phase acquired by the qubit during its evolution """
        def schrodinger(t, psi):
            return -1j * self.hamiltonian_matrix() @ psi
        
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval)
        
        berry_phase = 0.0
        for i in range(1, len(sol.t)):
            dt = sol.t[i] - sol.t[i-1]
            psi_prev = sol.y[:, i-1]
            psi_curr = sol.y[:, i]
            berry_phase += np.angle(np.vdot(psi_prev, psi_curr))
        
        return berry_phase

    def plot_berry_phase(self, psi0, t_span):
        """ Plot the Berry phase acquired by the qubit """
        berry_phase = self.calculate_berry_phase(psi0, t_span)
        plt.figure(figsize=(8, 6))
        plt.plot(t_span, [berry_phase]*len(t_span), label='Berry Phase')
        plt.xlabel('Time')
        plt.ylabel('Berry Phase')
        plt.title('Berry Phase Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
