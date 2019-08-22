"""
Dette er et test script til at tjekke om pakken fungerer
"""
#%%
import tomsup

## Get a list of valid agents
tomsup.valid_agents()

## tomsup - create a single agent
    # There is two ways to setup an agent (these are equivalent)
sirRB = tomsup.Agent(strategy = "RB", save_history = True, bias = 0.7)
sirRB = tomsup.RB(bias = 0.7, save_history = True)
isinstance(sirRB, tomsup.Agent)  # sirRB is an agent 
type(sirRB)  # of supclass RB

choice = sirRB.compete()



print(f"SirRB chose {choice} and his probability for choosing 1 was {sirRB.bias}.")

## Create a group of agents
desired_agent = ['RB']*3 + ['WSLS']*3  # list of desired agent strategies
print(desired_agent)
parameters = [{'bias': 0.7}]*3 + [{}]*3  # list of their starting parameters in dictionary format
print(parameters)

group = tomsup.create_agents(desired_agent, start_params = parameters)
print(type(group))

# inspect group
print(group.get_names())

## Set environment
group.set_env('round_robin')
print(group.get_environment_name())
print(group.get_environment())

##
out = group.compete(n_rounds = 10, n_sim = 1, p_matrix = 'penny_competitive')

#%%

#%%
