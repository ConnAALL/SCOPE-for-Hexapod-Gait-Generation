"""
Supervisor script for running CMA-ES with the new SCOPE controller.
This script manages the simulation environment and uses CMA-ES to evolve candidate controllers.
Each candidate is evaluated for a fixed number of simulation steps (TRIAL_STEPS),
and the robot is reset after each trial.
"""

import os
import sys
import json
import numpy as np

from SimulationControl import SimulationControl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SteadyStateGA import SteadyStateGA

config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.txt'))
with open(config_file, 'r') as file: config = json.load(file)

# SSGA hyperparameters
CHROMOSOME_SIZE = config["CHROMOSOME_SIZE"]
POP_SIZE = config["POP_SIZE"]
MAX_GENERATIONS = config["MAX_GENERATIONS"]
TRIAL_STEPS = config["TRIAL_STEPS"]
MUTATION_RATE = config["MUTATION_RATE"]
MUTATION_SCALE = config["MUTATION_SCALE"]
STABLE_FITNESS = bool(config["STABLE_FITNESS"])
AUTO_RESTART = config["AUTO_RESTART"]

# Parameters for logging - load all config values for complete parameter tracking
PARAMETERS = {
    "INPUT_DIMS": config["INPUT_DIMS"],
    "OUTPUT_SIZE": config["OUTPUT_SIZE"],
    "CHROMOSOME_SIZE": CHROMOSOME_SIZE,
    "POP_SIZE": POP_SIZE,
    "MAX_GENERATIONS": MAX_GENERATIONS,
    "TRIAL_STEPS": TRIAL_STEPS,
    "MUTATION_RATE": MUTATION_RATE,
    "MUTATION_SCALE": MUTATION_SCALE,
    "LIMIT_ROM_DEGREES": config["LIMIT_ROM_DEGREES"],
    "MOTOR_LIMITS": {name: tuple(bounds) for name, bounds in config['MOTOR_LIMITS_DEG'].items()},
    "STABLE_FITNESS": STABLE_FITNESS,
    "MODEL_TYPE": config["MODEL_TYPE"],
}

def run_steady_state_ga():
    """
    Main loop for the steady state genetic algorithm for real-time learning.
    Uses the SteadyStateGA class for population management.
    Each candidate chromosome is evaluated for a fixed number of simulation steps.
    """

    # Initialize the simulation control
    sim_control = SimulationControl()

    # Create the steady state GA
    ga = SteadyStateGA(
        chromosome_size=CHROMOSOME_SIZE,
        population_size=POP_SIZE,
        mutation_rate=MUTATION_RATE,
        mutation_scale=MUTATION_SCALE,
    )

    # Log the parameters of the simulation to a .txt file
    sim_control.log_parameters(PARAMETERS)

    def fitness_func(chromosome):
        """Main fitness function in the simulation"""

        # Send the chromsome to the mantis robot. 
        # Chromosome is the weights of the SCOPE controller.
        sim_control.send_message(chromosome.tolist())

        # Run trials for a fixed number of steps
        for _ in range(TRIAL_STEPS):
            sim_control.simulation_step()

        ### When the trial ends ###
        
        # Compute the distance difference from the last time it was taken
        distance_difference = sim_control.get_position_change()

        # Send a message to the robot to indicate that the trial ended
        sim_control.send_message(["TRIAL_END"])

        # Get the stability ratio from the mantis robot
        stability_ratio = sim_control.wait_message()

        # Depending on the vlaue in the config file, calculate th fitness as
        # The distance difference or the distance difference multiplied by the stability ratio
        if STABLE_FITNESS:
            fitness = distance_difference * stability_ratio
        else:
            fitness = distance_difference

        x, y, _ = sim_control.get_position()
        dist_to_center = np.sqrt(x**2 + y**2)
        
        if dist_to_center > 9000: 
            _, _, z = sim_control.get_position()
            sim_control.set_position([0, 0, z])
            sim_control.previous_x, sim_control.previous_y = 0, 0
        

        return distance_difference, stability_ratio, fitness

    # Evaluating the initial population
    for i in range(POP_SIZE):
        d, s, f = fitness_func(ga.population[i])
        ga.fitnesses[i] = f

        print(f"Initial evaluation: Individual {i+1}/{POP_SIZE}, Fitness: {f:.3f}")
        
        # For each initial individual, log the fitness values
        sim_control.log_fitness(
            generation=0,
            solution=i+1,
            fitness=f,
            distance_difference=d,
            stability_ratio=s
        )

    # The previous best fitness is none
    prev_best = None

    # The main training loop
    for generation in range(1, MAX_GENERATIONS + 1):
        
        # Produce two offsprings from the subpopulation
        offspring = ga.produce_offspring()
        offspring_metrics = []  # Will hold the distance and the stability ratios for the offsprings
        offspring_fitness  = []  # Will hold the fitness values for the offsprings

        # For each child, evaluate the offspring,
        # Calculate the fitness and save the metrics
        for child in offspring:
            d, s, f = fitness_func(child)
            offspring_metrics.append((d, s))
            offspring_fitness.append(f)

        # Replace the worst individuals in the subpopulation with these offsprings
        updated, replaced_idxs = ga.update_subpopulation(
            offspring,
            offspring_fitness,
            always_replace=True
        )

        # If there is an update
        if updated:
            # Get the distance and stability values for these offsprings
            (d0, s0), (d1, s1) = offspring_metrics

            # Print for the logging purposes in console
            print(f"[Generation {generation}]: "
                  f"Offspring with fitness {offspring_fitness[0]:.3f} "
                  f"replaced individual at index {replaced_idxs[0]}")
            print(f"                  "
                  f"Offspring with fitness {offspring_fitness[1]:.3f} "
                  f"replaced individual at index {replaced_idxs[1]}")

            # Log the fitness, distance change, and stability ratio for the replaced individuals
            sim_control.log_fitness(
                generation=generation,
                solution=replaced_idxs[0] + 1,
                fitness=offspring_fitness[0],
                distance_difference=d0,
                stability_ratio=s0
            )
            sim_control.log_fitness(
                generation=generation,
                solution=replaced_idxs[1] + 1,
                fitness=offspring_fitness[1],
                distance_difference=d1,
                stability_ratio=s1
            )
        else:
            print(f"[Generation {generation}]")

        # If there is a new best individual, save it
        best_candidate, best_fitness = ga.get_best_individual()
        if prev_best is None or not np.array_equal(best_candidate, prev_best):
            prev_best = best_candidate.copy()
            best_filename = f"{sim_control.timestamp}_Generation{generation}_BestIndv.npy"
            best_filepath = os.path.join(sim_control.log_dir, "BEST_INDV", best_filename)
            np.save(best_filepath, best_candidate)
            print(f"[Generation {generation}]: New best individual with fitness: {best_fitness:.3f}")
        
        if AUTO_RESTART > 0 and generation == AUTO_RESTART:
            print(f"[Generation {generation}]: Restarting simulation...")
            sim_control.supervisor.worldReload()


def main():
    run_steady_state_ga()

if __name__ == "__main__":
    main()
