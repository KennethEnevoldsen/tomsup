# Imports
import os
import random
import numpy as np
from time import time

import sys
sys.path.append('/Users/au561649/Desktop/Github/tomsup/python package')
sys.path.append('/Users/ptwaade/Desktop/Uni/tomsup/tomsup_package')

import tomsup as ts

#P1 forced choices:
forced_choices_p1 = [0, 1, 0, 1, 0, 1, 1, 1, 1, 0]
#P2 forced choices: 
forced_choices_p2 = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]

# Simulation settings
n_trials = 10

# Get payoff matrix
penny_comp = ts.PayoffMatrix(name='penny_competitive')

# Create agents
tom_1 = ts.TOM(level = 2, volatility= -2,  b_temp= -1,
               bias = 0, dilution = None, save_history = True)

tom_2 = ts.TOM(level = 2, volatility= -2,  b_temp= -1,
               bias = 0, dilution = None, save_history = True)

# Reset agents before start
tom_1.reset()
tom_2.reset()

# Set choices to None in the first roun
prev_choice_p1 = None
prev_choice_p2 = None

# Go through each trial one at a time
for trial in range(1, n_trials):
    # Make choices
    choice_1 = tom_1.compete(p_matrix=penny_comp, agent=0, 
                            op_choice=prev_choice_p1)
    choice_2 = tom_2.compete(p_matrix=penny_comp, agent=1,
                            op_choice=prev_choice_p2)

    # Save choices for next round
    prev_choice_1tom = forced_choices_p1[trial]
    prev_choice_2tom = forced_choices_p2[trial]