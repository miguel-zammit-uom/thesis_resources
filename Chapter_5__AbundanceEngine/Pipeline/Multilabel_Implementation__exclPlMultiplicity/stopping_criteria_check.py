import numpy as np


class StoppingCriteriaCheck:

    def __init__(self, first_pareto_front_size, current_gen_avg_fitness, prev_gen_avg_fitness, max_pop_size,
                 imputation_thresh, clf_thresh):
        """Check 3 of the 4 stopping criteria, returns True if (at least) one condition is met:
                1. All the population is non-dominated, ie. in the first pareto front
                2. Current and previous generation have an average imputation objective value smaller than threshold.
                3. Current and previous generation have an average classification objective value smaller than threshold
                """
        self.stop_run, self.criteria_met = False, 'None'

        if first_pareto_front_size >= max_pop_size:
            self.stop_run, self.criteria_met = True, '1. All population is non-dominated.'
        elif np.abs(current_gen_avg_fitness['Imputation'] - prev_gen_avg_fitness['Imputation']) <= imputation_thresh:
            self.stop_run, self.criteria_met = True, '2. Imputation threshold exceeded.'
        elif np.abs(current_gen_avg_fitness['Classification'] - prev_gen_avg_fitness['Classification']) <= clf_thresh:
            self.stop_run, self.criteria_met = True, '3. Classification threshold exceeded.'
