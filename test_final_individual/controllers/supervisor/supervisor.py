"""
Test script for testing the best SCOPE controller
"""

from controller import Supervisor  # type:ignore
import numpy as np

from SimulationControl import SimulationControl

# Update this path to point to your best individual's .npy file.
BEST_INDIVIDUAL_FILE = "bestIndv.npy"

def main():
    # Create the simulation control instance.
    sim_control = SimulationControl()

    # Load the best individual from the .npy file.
    best_chromosome = np.load(BEST_INDIVIDUAL_FILE)

    print("Loaded best individual from file:", BEST_INDIVIDUAL_FILE)

    # Send the best chromosome to the robot.
    sim_control.send_message(best_chromosome.tolist())
    print("Best individual sent to the robot.")

    while True:
        sim_control.simulation_step()

if __name__ == "__main__":
    main()
