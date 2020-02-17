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

ts.PayoffMatrix("penny_competitive")

ts.create_agents