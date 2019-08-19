from warnings import warn

#%% Decision functions


def basic_p_op_1_fun ():
    pass

def p_op_1_k_fun():
    pass

def expected_payoff_difference (p_op, agent, p_matrix):
    """
    Calculate expected payoff of choosing 1 over 0
    """
    return (p_op * (p_matrix.outcome(1, 1, agent) - p_matrix.outcome(0, 1, agent)) + 
           (1 - p_op) * (p_matrix.outcome(1, 0, agent) - p_matrix.outcome(0, 0, agent)))

def softmax (e_pay_diff, b_temp): 
    """
    Softmax function for calculating own probability of choosing 1
    """
    #Input variable transforms
    b_temp = exp(b_temp)

    #Calculation
    p_self = 1 / (1 + exp(-(e_pay_diff / b_temp)))

    #Set output bounds
    if p_self > 0.999:
        p_self = 0.999
        warn("Choice probability constrained at upper bound 0.999 to avoid rounding errors", Warning)
    if p_self < 0.001:
        p_self = 0.001
        warn("Choice probability constrained at lower bound 0.001 to avoid rounding errors", Warning)

    return p_self

def decision_function ()



#%% Learning functions
def c_var0_update (prev_var0, prev_mean0, volatility):
    """ 
    0-ToM updates variance / uncertainty on choice probability estimate
    """
    #Input variable transforms
    volatility = exp(volatility)
    prev_var0 = exp(prev_var0)
    prev_mean0 = inv_logit(prev_mean0)

    #Update
    new_var0 = 
       1 / (
       (1 / (volatility + prev_var0)) +
       prev_c_mean0 * (1 - prev_mean0))

    #Output variable transform   
    new_var0 = log(new_var0)

    return new_var0

def c_mean0_update (prev_mean0, var0, op_choice):
    """
    0-ToM updates mean choice probability estimate
    """
    #Input variable transforms
    var0 = exp(var0)
    prev_mean0 = inv_logit(prev_mean0)
    
    #Update
    new_mean0 =
         prev_mean0 + var0 * (op_choice - prev_mean0)
    
    return new_mean0

def p_op_k_approx_fun (prev_mean, prev_var, prev_gradient, level):
    """
    Approximates the estimated choice probability of the opponent on the previous round. 
    A semi-analytical approximation derived in Daunizeau, J. (2017)
    """

    #Constants
    a = 0.205
    b = -0.319
    c = 0.781
    d = 0.870

    #Input variable transforms
    prev_var = exp(prev_var)

    #Prepare variance by weighing with gradient
    prev_var_prepped = []
    for level_index in range(level):
        prev_var_prepped[level_index] = prev_var[level,:].T.dot(prev_gradient[level,:]**2) 

    #Equation
    p_op_approx = inv_logit (
        (prev_mean + b * prev_var_prepped^c) / sqrt(1 + a prev_var_prepped^d))

    #Output variable transform
    p_op_approx = log(p_op_approx)

    return = p_op_approx

def pk_udpate(prev_pk, p_op_approx, dilution = None, op_choice):


    if dilution:
        print(dilution)

    #Input variable transforms


def parameter_variance_update ():

def parameter_mean_update ():

def gradient_update ():

def rec_learning_function ():



#%% Other functions

def k_tom ():

def rec_prepare_k_tom ():

def logit (p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))
