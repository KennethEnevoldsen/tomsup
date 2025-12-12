# This script is used to compare the python implementation
# with the matlab implementation step-by-step. We suggest
# using a debugger for this. Note that the VBA package
# for matlab needs to be added to the matlab path for the
# matlab scripts to work

# Imports
import sys

sys.path.append("/Users/au568658/Desktop/Academ/Projects/tomsup")


import tomsup as ts

# P1 forced choices:
forced_choices_p1 = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
# P2 forced choices:
forced_choices_p2 = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0]

# Simulation settings
n_trials = 10

# Get payoff matrix
penny_comp = ts.PayoffMatrix(name="penny_competitive")

# Create agents
player_1 = ts.RB()

player_2 = ts.TOM(
    level=1, volatility=-2, b_temp=-1, bias=0, dilution=None, save_history=True
)

# Reset agents before start
player_1.reset()
player_2.reset()

# Go through each trial one at a time
for trial in range(n_trials):
    print(trial)

    # Save opponent choices and save
    prev_choice_p1 = forced_choices_p1[trial]
    prev_choice_p2 = forced_choices_p2[trial]

    # Make choices
    choice_1 = player_1.compete(p_matrix=penny_comp, agent=0, op_choice=prev_choice_p2)
    choice_2 = player_2.compete(p_matrix=penny_comp, agent=1, op_choice=prev_choice_p1)

    # Reset own choices
    player_1.choice = forced_choices_p1[trial + 1]
    player_2.choice = forced_choices_p2[trial + 1]

    print("SELF: {}".format(prev_choice_p2))
    print("OPPONENT: {}".format(prev_choice_p1))
    player_2.print_internal()
