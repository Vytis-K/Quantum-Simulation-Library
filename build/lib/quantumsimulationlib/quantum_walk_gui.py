import tkinter as tk
from tkinter import messagebox
import numpy as np
from quantum_walk import QuantumWalk

class QuantumCoinGUI:
    def __init__(self, master):
        self.master = master
        master.title("Quantum Coin Generator")

        self.label = tk.Label(master, text="Enter your custom coin matrix (2x2):")
        self.label.pack()

        self.entry11 = tk.Entry(master, width=5)
        self.entry11.pack()
        self.entry12 = tk.Entry(master, width=5)
        self.entry12.pack()
        self.entry21 = tk.Entry(master, width=5)
        self.entry21.pack()
        self.entry22 = tk.Entry(master, width=5)
        self.entry22.pack()

        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack()

        self.walk_button = tk.Button(master, text="Run Quantum Walk", command=self.run_walk)
        self.walk_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.quantum_walk = None

    def submit(self):
        try:
            # Retrieve matrix entries from GUI
            a = complex(self.entry11.get())
            b = complex(self.entry12.get())
            c = complex(self.entry21.get())
            d = complex(self.entry22.get())
            matrix = np.array([[a, b], [c, d]])
            
            # Set this matrix as the coin operation
            self.quantum_walk = QuantumWalk(num_positions=10, start_position=5, coin_operation=matrix)
            messagebox.showinfo("Success", "Matrix set successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            self.quantum_walk = None

    def run_walk(self):
        if self.quantum_walk is None:
            messagebox.showerror("Error", "No valid quantum walk initialized!")
        else:
            self.quantum_walk.step()
            probabilities = self.quantum_walk.measure()
            self.result_label.config(text=f"Probability distribution: {probabilities}")

def main():
    root = tk.Tk()
    gui = QuantumCoinGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
