"""
docstring

TODO: figure out hvad vi gør med payoff matrices
"""

import json
import numpy as np

agent_dict = {
    'RB': {
     'name': 'Random Bias',
     'shorthand': 'RB',
     'example': 'RB(bias = 0.7)', 
     'reference': 'Devaine, et al. (2017)', 
     'strategy': 'Chooses 1 randomly based on a probability or bias'}, 
    'WSLS': {
     'name': 'Win-stay, lose-switch',
     'shorthand': 'WSLS',
     'example': 'WSLS()', 
     'reference': 'Nowak & Sigmund (1993)', 
     'strategy': 'If it win it chooses the same option again, if it lose it change to another'},
    'TFT': {
     'name': 'Tit-for-Tat'
     'shorthand': 'TFT',
     'example': 'TFT()',
     'reference': 'Shelling (1981)',
     'strategy': 'Intended the prisoners dilemma. It starts out cooperating and then simply copies it opponents action.'},
    } 
    
    }


with open('../siptom/agent_info.json', 'w') as fp:
    json.dump(agent_dict, fp)

with open('agent_info.json', 'r') as fp:
    data = json.load(fp)




