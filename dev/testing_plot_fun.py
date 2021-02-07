"""
"""

import os
os.chdir("/Users/au561649/Desktop/Github/tomsup/python package/")
os.getcwd()
import tomsup as ts

agents = ['RB', '1-ToM', '2-ToM', 'WSLS']
start_params = [{'bias': 0.7}, {}, {}, {}]

group = ts.create_agents(agents, start_params)

group.set_env(env='round_robin')
penny = ts.PayoffMatrix(name="penny_competitive")
group.compete(p_matrix=penny, n_rounds=20, n_sim=4,
              save_history=True)

df = group.get_results()

group.plot_heatmap()

df = ts.ResultsDf(df)


ts.plot_heatmap