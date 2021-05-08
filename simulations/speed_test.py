# Imports
import os
import random
import numpy as np
from time import time

import sys
sys.path.append('/Users/au561649/Desktop/Github/tomsup/python package')
sys.path.append('/Users/ptwaade/Desktop/Uni/tomsup/tomsup_package')

import tomsup as ts

# Set seed
random.seed(1995)

# - Simulation settings - #
n_tests = 20
n_sim = 5
n_rounds = 60
#(Short run)
n_tests = 2
n_sim = 2
n_rounds = 10

# Get payoff matrix
penny_comp = ts.PayoffMatrix(name='penny_competitive')

# Create list of agents
agents = ['RB', '2-ToM']
# Set parameters
start_params = [{}, {}]

# Initialize vector for populaitng with times
elapsed_times = [None] * n_tests

for test in range(n_tests):

    print(test)

    # Get start time
    start_time = time()

    # Make group
    group = ts.create_agents(agents, start_params)

    # Set as round robin tournament
    group.set_env(env='round_robin')

    # Run tournament
    results = group.compete(p_matrix=penny_comp, n_rounds=n_rounds,
                            n_sim=n_sim, save_history=False, verbose=False)

    # Save elapsed time in vector
    elapsed_times[test] = time() - start_time


# Get mean and standard deviations
print(np.mean(elapsed_times))
print(np.std(elapsed_times))





