#Using the invlogit and logit from VBA, with finite precision
#"Fornumerical Purposes" in Decision function, in param mean update both for 0 and k.
#Fixed the way bias was implemented
#Fixed param_var_update input, and made sure the matreces are multiplied together correctly
#Made bias gradient prior = 0.999999997998081, like in the VBA package

import tomsup as ts
from warnings import warn
import numpy as np
import copy
#from scipy.special import expit as inv_logit
#from scipy.special import logit

penny = ts.PayoffMatrix(name='penny_competitive')

#arnault = ts.TOM(level=3, save_history=True)
arnault = ts.TOM(level=2, save_history=True)
#arnault = ts.TOM(level=1, save_history=True)
#arnault= ts.TOM(level=0, save_history=True)

print('--- ROUND 1 ----')
# runde 1
arnault.compete(p_matrix=penny, agent=0, op_choice=None)
arnault.choice = 1
arnault.print_internal()

print('--- ROUND 2 ----')
#runde 2
arnault.compete(p_matrix=penny, agent=0, op_choice=1)
arnault.choice = 1
arnault.print_internal()

print('--- ROUND 3 ----')
# runde 3
arnault.compete(p_matrix=penny, agent=0, op_choice=1)
arnault.choice = 1
arnault.print_internal()

print('--- ROUND 4 ----')
# runde 4
arnault.compete(p_matrix=penny, agent=0, op_choice=1)
arnault.choice = 1
arnault.print_internal()

print('--- ROUND 5 ----')
# runde 5
arnault.compete(p_matrix=penny, agent=0, op_choice=1)
arnault.choice = 1
arnault.print_internal()

print('--- ROUND 6 ----')
# runde 6
arnault.compete(p_matrix=penny, agent=0, op_choice=1)
arnault.choice = 1
arnault.print_internal()

print('--- ROUND 7 ----')
# runde 7
arnault.compete(p_matrix=penny, agent=0, op_choice=1)
arnault.choice = 1
arnault.print_internal()


# for i in range(20):
#     print(i)
#     # runde 7
#     arnault.compete(p_matrix=penny, agent=0, op_choice=1)
#     arnault.choice = 1
#     arnault.print_internal()

# arnault.get_history()



#%%
# ######## Starting line-by-line ################

# prev_internal_states = arnault.internal
# params = arnault.params
# self_choice = arnault.choice
# op_choice = 1
# level = arnault.level
# agent = 0
# p_matrix = penny


# #Extract needed parameters
# volatility = params['volatility']
# if 'dilution' in params:
#     dilution = params['dilution']
# else:
#     dilution = None

# #Make empty dictionary for storing updates states
# new_internal_states = {}
# opponent_states = {}

# #Extract needed variables
# prev_p_k = prev_internal_states['own_states']['p_k']
# prev_p_op_mean = prev_internal_states['own_states']['p_op_mean']
# prev_param_mean = prev_internal_states['own_states']['param_mean']
# prev_param_var = prev_internal_states['own_states']['param_var']
# prev_gradient = prev_internal_states['own_states']['gradient']

# #Update opponent level probabilities
# p_opk_approx = p_opk_approx_fun(prev_p_op_mean, prev_param_var, prev_gradient, level)
# p_k = p_k_udpate(prev_p_k, p_opk_approx, op_choice, dilution)


# #Update parameter estimates
# param_var = param_var_update(prev_p_op_mean, prev_param_var, prev_gradient, p_k, volatility)
# param_mean = param_mean_update(prev_p_op_mean, prev_param_mean, prev_gradient, p_k, param_var, op_choice)

# #Prepare perspective
# p_op_mean = np.zeros(level)
# gradient = np.zeros([level, param_mean.shape[1]])

# sim_agent = 1 - agent #simulated perspective swtiches own and opponent role
# sim_self_choice, sim_op_choice = op_choice, self_choice #And previous choices

# #for sim_level in range(level):
# sim_level = 0

# #Further preparation of simulated perspective
# sim_prev_internal_states = copy.deepcopy(prev_internal_states['opponent_states'][sim_level])

# #Make parameter structure similar to own
# sim_params = copy.deepcopy(params)

# #Populate it with estimated values
# for param_idx, param_key in enumerate(params):
#     sim_params[param_key] = param_mean[sim_level, param_idx]

# #Simulate opponent learning (recursive)
# sim_new_internal_states = learning_function(
#     sim_prev_internal_states,
#     sim_params,
#     sim_self_choice,
#     sim_op_choice,
#     sim_level,
#     sim_agent,
#     p_matrix)

# #Simulate opponent deciding
# p_op_mean[sim_level] = decision_function(
#     sim_new_internal_states,
#     sim_params,
#     sim_agent,
#     sim_level,
#     p_matrix)

# #Update gradient (recursive)
# gradient[sim_level] = gradient_update(
#     params,
#     p_op_mean[sim_level],
#     param_mean[sim_level],
#     sim_prev_internal_states,
#     sim_self_choice,
#     sim_op_choice,
#     sim_level,
#     sim_agent,
#     p_matrix)
