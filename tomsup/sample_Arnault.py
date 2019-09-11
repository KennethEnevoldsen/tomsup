"""
A sample for Arnault. Should work with the current distribution.

There is still minor bugs which needs to be fixed. 
We will update you tomorrow regarding these, 
but the current distribution should be enough to get everything working.


me and peter will be at the library so feel free to ask us.
"""

import tomsup as ts

# Create agent
arnault = ts.Agent(strategy="2-tom", save_history=True)
# or
arnault = ts.TOM(level=2, save_history=True)

# you can see all its internal states like this (maybe not relevant unless you know the model)
arnault.print_internal()

# define payoffmatrix for the game
penny = ts.PayoffMatrix(name='penny_competitive')
# this is simply a 2x2x2 matrix as you see:
print(penny())

n_rounds = 10
p_choice = None
for round in range(n_rounds): 
    tom_choice = arnault.compete(p_matrix=penny, agent=0, op_choice=p_choice)  # agent is the perspective in the p_matrix
    
    p_choice = input()  # get the input from the human player (up to you to visualize this)
    
    # calculate the players payoff
    players_payoff = penny.payoff(action_agent0=tom_choice, action_agent1=p_choice, agent=1)

    # save the output of the round
    
# get the internal states of the agent
history = arnault.get_history()  # NOTE! This should only be for the agents internal states as this does NOT record the opponents action.
history.to_json(r'Name.json')  # JSON as the internal states of the TOM agent is structured as a tree


