"""
Dette er et testscript til at tjekke om pakken fungerer
"""

#Using the invlogit and logit from VBA, with finite precision
#"Fornumerical Purposes" in Decision function, in param mean update both for 0 and k.
#Fixed the way bias was implemented
#Fixed param_var_update input (p_op_mean instead of param_mean), and made sure the matreces are multiplied together correctly
#Made bias gradient prior = 0.999999997998081, like in the VBA package

#Import packages
import os
os.chdir('..')
import tomsup as ts
import random
import numpy as np
from scipy.special import expit as inv_logit
from scipy.special import logit as logit

#Set seed for reporoducibility
random.seed(2)

#Simulation settings
n_sim = 2
#n_sim = 200
n_rounds = 2
#n_rounds = 100

#Get payoff matrix
penny_comp = ts.PayoffMatrix(name='penny_competitive')

#Create list of agents
all_agents = ['RB', 'WSLS', 'QL', 
          '0-TOM', '1-TOM', '2-TOM', '3-TOM', '4-TOM', '5-TOM']

#Write down parameter means
params_means = [0.8, 0.9,0.9, 0.5, -2,-1, -2,-1, -2,-1, -2,-1, -2,-1, -2,-1]
#And the variances of each mean (in this case all the same)
params_vars = [0.1]*len(params_means)

#Make empty list for inserting parameter values
parvals = [0]*len(params_means)

#For each simulation
for sim in range(n_sim):

    #Resample parameter values
    for idx, mean in enumerate(params_means):
        #The first five parameters are probability parameters
        if idx <= 4:
            #So they have to be constrained between 0 and 1 by a logit-inv_logit transform
            parvals[idx] = inv_logit(np.random.normal(logit(mean), params_vars[idx]))
        #But the other parameters
        else:
            #Can just be sampled
            parvals[idx] = np.random.normal(mean, params_vars[idx])
    
    #Save them for group input
    all_params = [{'bias': parvals[0]}, {'prob_stay': parvals[1], 'prob_switch': parvals[2]}, {'learning_rate': parvals[4]}, 
                    {'volatility':parvals[5], 'b_temp':parvals[6]}, {'volatility':parvals[7], 'b_temp':parvals[8]},
                    {'volatility':parvals[9], 'b_temp':parvals[10]}, {'volatility':parvals[11], 'b_temp':parvals[12]},
                    {'volatility':parvals[13], 'b_temp':parvals[14]}, {'volatility':parvals[15], 'b_temp':parvals[16]}]

    #Add save_history to all parameter sets
    for d in all_params:
        d['save_history'] = True

    #And remake the group
    group = ts.AgentGroup(all_agents, all_params)
    group.set_env(env = 'round_robin')
 
    #If its the first simulation
    if sim == 0:
        #Do the tournament and initate the results dataframe
        results = group.compete(p_matrix = penny_comp, n_rounds = n_rounds, save_history = True)
        #Add column with simulation number
        results['n_sim'] = sim
    #Otherwise
    else:
        #Run the tournament again
        result_onesim = group.compete(p_matrix = penny_comp, n_rounds = n_rounds, save_history = True)
        #Add column with simulation number
        result_onesim['n_sim'] = sim

        #And append to the results dataframe
        results = results.append(result_onesim, ignore_index = True)

#Save to CSV and pkl
results.to_csv(r'Large_Simulation_results.csv')
results.to_pickle(r'Large_Simulation_results.pkl')

#Examine the first 5 rows in results
results.head(20) 

