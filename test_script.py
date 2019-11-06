"""
Dette er et testscript til at tjekke om pakken fungerer
"""

#Using the invlogit and logit from VBA, with finite precision
#"Fornumerical Purposes" in Decision function, in param mean update both for 0 and k.
#Fixed the way bias was implemented
#Fixed param_var_update input, and made sure the matreces are multiplied together correctly
#Made bias gradient prior = 0.999999997998081, like in the VBA package

import tomsup as ts

penny = ts.PayoffMatrix(name='penny_competitive')

arnault = ts.TOM(level=2, save_history=True)
bond = ts.TOM(level=1, save_history=True)

results = ts.compete(arnault, bond, penny, n_rounds = 50, n_sim = 1000)

results['payoff_agent0'].mean()

arnault.get_history()['internal_states'][29]
bond.get_history()['internal_states'][29]

arnault.get_history(key = 'internal_states')

