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


