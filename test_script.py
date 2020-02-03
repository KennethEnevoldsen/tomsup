"""
Dette er et testscript til at tjekke om pakken fungerer
"""

#Using the invlogit and logit from VBA, with finite precision
#"Fornumerical Purposes" in Decision function, in param mean update both for 0 and k.
#Fixed the way bias was implemented
#Fixed param_var_update input (p_op_mean instead of param_mean), and made sure the matreces are multiplied together correctly
#Made bias gradient prior = 0.999999997998081, like in the VBA package

#Import packages
import tomsup as ts
import random

#Set seed for reporoducibility
random.seed(2)

#Simulation settings
n_sim = 10
n_rounds = 50

#Get payoff matrix
penny_comp = ts.PayoffMatrix(name='penny_competitive')

#Create list of agents
all_agents = ['RB', 'WSLS', 'QL', 
          'TOM', 'TOM', 'TOM', 'TOM', 'TOM', 'TOM']
#And give them parameters (default if no input is given
all_params = [{'bias': 0.8}, {'prob_stay' = 0.9, 'prob_switch' = 0.9}, {'learning_rate': 0.5}, 
                {'level' = 0}, {'level' = 1}, {'level' = 2}, {'level' = 3}, {'level' = 4}, {'level' = 5}]

#Create the agent group
group = ts.AgentGroup(all_agents, all_params)
#Set the environment to a round robin tournament
group.set_env(env = 'round_robin')

#Print the full group
print(group)
#Make a space for the coming outputs
print("\n----\n") 

#Hold the toiurnament
results = group.compete(p_matrix = penny, n_rounds = n_rounds, n_sim = n_sim)
#Examine the first 5 rows in results
results.head() 





arnault = ts.TOM(level=2, save_history=True)
bond = ts.TOM(level=1, save_history=True)

results = ts.compete(arnault, bond, penny_comp, n_rounds = n_rounds, n_sim = n_sim)

results['payoff_agent0'].mean()

arnault.get_history()['internal_states'][29]
bond.get_history()['internal_states'][29]

arnault.get_history(key = 'internal_states')
