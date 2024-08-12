import numpy as np


def flatten(l):
    return [item for sublist in l for item in sublist]

# Function from https://github.com/adam-katona/NSGA_2_tutorial/blob/master/NSGA_2_tutorial.ipynb
# amended slightly to factor for the inclusion of nan fitnesses.
def _calculate_domination_matrix(fitnesses):
    pop_size = fitnesses.shape[0]
    num_objectives = fitnesses.shape[1]
    fitness_grid_x = np.zeros([pop_size, pop_size, num_objectives])
    fitness_grid_y = np.zeros([pop_size, pop_size, num_objectives])
    for i in range(pop_size):
        fitness_grid_x[i, :, :] = fitnesses[i]
        fitness_grid_y[:, i, :] = fitnesses[i]
    # Set all nan values to -infinity so as to never dominate
    fitness_grid_x[np.isnan(fitness_grid_x)] = -np.inf
    fitness_grid_y[np.isnan(fitness_grid_y)] = -np.inf

    larger_or_equal = fitness_grid_x >= fitness_grid_y
    larger = fitness_grid_x > fitness_grid_y
    return np.logical_and(np.all(larger_or_equal, axis=2), np.any(larger, axis=2))


# Function from https://github.com/adam-katona/NSGA_2_tutorial/blob/master/NSGA_2_tutorial.ipynb
def _fast_calculate_pareto_fronts(fitnesses):
    # Calculate dominated set for each individual
    domination_sets = []
    domination_counts = []

    domination_matrix = _calculate_domination_matrix(fitnesses)
    pop_size = fitnesses.shape[0]

    for i in range(pop_size):
        current_dimination_set = set()
        domination_counts.append(0)
        for j in range(pop_size):
            if domination_matrix[i, j]:
                current_dimination_set.add(j)
            elif domination_matrix[j, i]:
                domination_counts[-1] += 1

        domination_sets.append(current_dimination_set)

    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts == 0)[0]
        if len(current_front) == 0:
            # print("Done")
            break
        # print("Front: ",current_front)
        fronts.append(current_front)

        for individual in current_front:
            domination_counts[
                individual] = -1 # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
            dominated_by_current_set = domination_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                domination_counts[dominated_by_current] -= 1

    return fronts


class TournamentSelection:

    def __init__(self, population, fitnesses, num_replacement_fronts):
        """Determine the Pareto solutions in the population. After this first Pareto front is found, it is removed from
        the population and the next Pareto front is found. This keeps on going till the population is sorted.

        The chromosomes in the first Pareto front will move to the next population with no change. The chromosomes in
        the last Pareto fronts (can be a GA hyperparameter) will be replaced by the crossover operator. The number of
        these chromosomes is defined approximately half of the chromosomes where are not in the first Pareto front."""

        self.population = population
        self.fitnesses = fitnesses
        self.num_replacement_fronts = num_replacement_fronts

        self.pareto_fronts = _fast_calculate_pareto_fronts(self.fitnesses)

        self.first_pareto_front = self.pareto_fronts[0]
        self.replacement_fronts = flatten(self.pareto_fronts[-num_replacement_fronts:])
        self.mutation_fronts = flatten(self.pareto_fronts[1:(len(self.pareto_fronts)-num_replacement_fronts)])
