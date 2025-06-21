"""
Main file to manage the SteadyState Genetic Algorithm (GA) for real-time learning.
"""

import numpy as np

class SteadyStateGA:
    def __init__(self,
                 chromosome_size,
                 population_size=20,
                 mutation_rate=0.1,
                 mutation_scale=0.1):
        """
        Create the Steady State Genetic Algorithm (GA) instance.
            - chromosome_size (int): The length of each candidate chromosome.
            - population_size (int): Number of individuals in the population.
            - mutation_rate (float): Probability of mutation per gene.
            - mutation_scale (float): Standard deviation for the Gaussian mutation.
        """
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

        # Initialize a random population
        self.population = [np.random.uniform(-1, 1, chromosome_size) for _ in range(population_size)]
        self.fitnesses = [0.0] * population_size  # Fitness values for each individual

    def crossover(self, parent1, parent2):
        """
        Produce two children (shape (2, chromosome_size)) via single‑point crossover.
        """
        cp = np.random.randint(1, self.chromosome_size)
        c1 = np.concatenate((parent1[:cp], parent2[cp:]))
        c2 = np.concatenate((parent2[:cp], parent1[cp:]))
        return np.stack([c1, c2], axis=0)

    def mutate(self, offspring):
        """
        Mutate the offspring by applying Gaussian noise with probability equal to the mutation rate.
        """
        mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
        offspring[mutation_mask] += np.random.normal(0, self.mutation_scale, np.sum(mutation_mask))
        return offspring

    def produce_offspring(self, subpop_size=5):
        """
        Generate a single new offspring using a sub-population.
        A random sub-population of size 'subpop_size' is selected and saved into the instance variable.
        The top two individuals (based on fitness) from this sub-population are used as parents.
        """
        # Select sub-population indices randomly from the full population.
        indices = np.random.choice(range(self.population_size), size=subpop_size, replace=False)
        self.current_subpop = indices  # Update instance variable with current subpopulation indices
        
        # Build a list of tuples: (index, individual, fitness)
        subpop = [(i, self.population[i], self.fitnesses[i]) for i in indices]
        # Sort the sub-population based on fitness (highest first)
        subpop_sorted = sorted(subpop, key=lambda x: x[2], reverse=True)

        # Create an off-spring using the top two individuals of the sub-population
        parent1 = subpop_sorted[0][1]
        parent2 = subpop_sorted[1][1]
        offspring = self.crossover(parent1, parent2)
        offspring = self.mutate(offspring)

        return offspring

    def update_subpopulation(self, offspring, offspring_fitness, always_replace=False):
        """
        Replace the worst individual within the current sub-population (stored in self.current_subpop).
        If the offspring is better than the worst individual, it replaces it.
        If always_replace is True, the offspring will replace the worst individual regardless of fitness.
        """
        # 1) find the two worst indices (lowest fitness) in the subpopulation
        sorted_by_fit = sorted(self.current_subpop, key=lambda idx: self.fitnesses[idx])
        worst_idxs = sorted_by_fit[:2]   # two indices with smallest fitness

        results = []
        changed = False
        # 2) for each offspring and its target worst slot, do the replacement test
        for offspring, fit, worst_idx in zip(offspring, offspring_fitness, worst_idxs):
            if always_replace or fit > self.fitnesses[worst_idx]:
                self.population[worst_idx] = offspring
                self.fitnesses[worst_idx] = fit
                results.append(worst_idx)
                changed = True
            else:
                results.append(worst_idx)
        return changed, results

    def get_best_individual(self):
        """
        Returns the best candidate in the population and its fitness.
        """
        best_idx = np.argmax(self.fitnesses)
        return self.population[best_idx], self.fitnesses[best_idx]
