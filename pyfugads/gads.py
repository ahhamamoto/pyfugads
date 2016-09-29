#!/usr/bin/env python3

import numpy as np
import pandas as pd


class GADS():
    """Class to generate the DSNSF from an input."""

    def __init__(self, dataset, pop_size, generations, crossover_rate,
                 mut_prob, mut_rate, selection=0):
        """Just defines the GA parameters."""
        self.dataset = np.transpose(dataset)
        self.population_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_probability = mut_prob
        self.mutation_rate = mut_rate
        self.selection_method = selection

        self.generate_population(0)
        self.generate_population(1)

    def generate_population(self, idx):
        """Generate initial population based on input."""
        inferior_limit = np.min(self.dataset[idx])
        superior_limit = np.max(self.dataset[idx])
        self.population = np.random.uniform(inferior_limit, superior_limit,
                                            self.population_size)
        print(self.population)
