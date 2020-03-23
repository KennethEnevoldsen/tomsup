#This is only relevant if tomsup was installed from the github folder
#Move up to the correct folder
import os
os.chdir('..')
os.chdir('..')

#And import the tomsup library
import tomsup as ts

#Create environment
penny = ts.PayoffMatrix("penny_competitive")

#Creating the agents
rb = ts.RB(bias = 0.8, #choice probability of RB
            save_history=False) #there are no states to save   

tom0 = ts.TOM(level = 0, #sophistication level
            volatility= -2, #assumption of volatility in opponent parameters
            b_temp= -1, #temperature - "randomness" og behavior
            save_history = True) #saving ToM's states on every trial

tom2 = ts.TOM(level = 2, 
            volatility= -2, 
            b_temp= -1, 
            bias = 0, #The agent is not biased, but can estimate biases
            dilution = None, #The agent does not forget and also does not estimate dilution
            save_history = True) #saving ToM's states on every trial

#Single trial choices
rb.compete(p_matrix = penny, #the payoff matrix used
            agent = 0, #agent 0 is the seeker (?)
            op_choice = None) #the opponent's choice is None on the first round. But RB doesn't care anyways.

tom0.compete(p_matrix = penny,
            agent = 0,
            op_choice = None) 

tom0.compete(p_matrix = penny, 
            agent = 0,
            op_choice = 0) #from the second round, ToM agents need to know the opponent's choice

#Make them play against each other
results = ts.compete(rb, tom0, p_matrix = penny, n_rounds = 30, save_history=True)

#Look at tom's internal states
tom0.get_history()["internal_states"][10]['own_states'] #they're in the agent
results["history_agent1"][10]["internal_states"]['own_states'] #but also in the results dataframe

#Make a group competition
agents = ['RB', '0-TOM', '2-TOM'] #Select agents
params = [{'bias': 0.7}, #RB's parameters
          {}, #0-ToM's parameters (empty means default)
          {'volatility': -3, 'b_temp': 0, 'dilution': 0.1}] #2-ToM's parameters

group = ts.create_agents(agents, params) #Create the group
group.set_env(env = 'round_robin') #Set the tournament structure. Right now there's only one option
print(group) #check the group settings

#Make the group compete
results = group.compete(p_matrix = penny, n_rounds = 20, n_sim = 4)
results.head() #examine the first 5 rows in results

#Plot score and choices for some competing agents (WORK IN PROGRESS)
ts.plot.score(results, agent0 = "RB", agent1 = "0-TOM", agent = 0)
ts.plot.choice(results, agent0 = "RB", agent1 = "2-TOM", agent = 0)

#It's also possible for people to play against agents from tomsup. We have made a basic psychopy script for this.