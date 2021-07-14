# Imports
import os
import random
import numpy as np
from time import time
import cProfile
import io
import pstats

print(os.getcwd())
print(os.listdir())

import sys

sys.path.append("/Users/au561649/Desktop/Github/tomsup/python package")

import tomsup as ts

# Set seed
random.seed(1995)

# - Simulation settings - #
n_tests = 50
n_sim = 10
n_rounds = 20

# Get payoff matrix
penny_comp = ts.PayoffMatrix(name="penny_competitive")

# Create list of agents
agents = ["RB", "3-ToM"]
# Set parameters
start_params = [{}, {}]

# Initialize vector for populaitng with times
elapsed_times = [None] * n_tests

pr = cProfile.Profile()
pr.enable()

for test in range(n_tests):

    print(test)

    # Get start time
    start_time = time()

    # Make group
    group = ts.create_agents(agents, start_params)

    # Set as round robin tournament
    group.set_env(env="round_robin")
    # Run tournament
    results = group.compete(
        
        p_matrix=penny_comp,
        n_rounds=n_rounds,
        n_sim=n_sim,
        save_history=False,
        verbose=False,
        n_jobs=-1,
    )

    # Save elapsed time in vector
    elapsed_times[test] = time() - start_time


# Get mean and standard deviations
print(np.mean(elapsed_times))
print(np.std(elapsed_times))

pr.disable()


with open("output_time2.txt", "w") as f:
    ps = pstats.Stats(pr, stream=f)
    ps.strip_dirs().sort_stats("time").print_stats()

with open("output_calls2.txt", "w") as f:
    ps = pstats.Stats(pr, stream=f)
    ps.strip_dirs().sort_stats("calls").print_stats()

# python -m cProfile [-o output_file] [-s sort_order] (-m module | myscript.py)
# python -m cProfile -o speed.txt  simulations/speedtest.py
