"""
"""

import os
os.chdir("/Users/au561649/Desktop/Github/tomsup/python package/")
os.getcwd()
import tomsup as ts

agents = ['RB', '1-ToM', '2-ToM']
start_params = [{'bias': 0.7}, {'learning_rate': 0.5}, {}]

group = ts.create_agents(agents, start_params)

group.set_env(env='round_robin')
penny = ts.PayoffMatrix(name="penny_competitive")
results = group.compete(p_matrix=penny, n_rounds=20, n_sim=4,
                        save_history=True)

isinstance(results, ts.ResultsDf)
score(df=results, agent0="RB", agent1="1-ToM", agent=0)
choice(df=results, agent0="RB", agent1="1-ToM", agent=0)
results


# HEATMAP
# PLOT 0 TOM og KTOM estimates
# estimated choid
# add plots to ResultsDF class
# remove plot_results