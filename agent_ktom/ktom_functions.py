from warnings import warn
import numpy as np

#%% Decision functions

def p_op0_fun (p_op_mean0, p_op_var0):
    """
    0-ToM combines the mean and variance of its parameter estimate into a final choice probability estimate.
    NB: This is the function taken from the VBA package (Daunizeau 2014), which does not use 0-ToM's volatility parameter to avoid unidentifiability problems.
    """
    #Constants
    a = 0.36

    #Input variable transforms
    p_op_var0 = np.exp(p_op_var0)

    #Calculate
    p_op0 = p_op_mean0 / np.sqrt(1 + a * p_op_var0)

    #Output variable transforms
    p_op0 = inv_logit(p_op0)

    return p_op0

def p_opk_fun (p_op_mean, param_var, gradient):
    """
    k-ToM combines the mean choice probability estimate and the variances of its parameter estimates into a final choice probability estimate.
    NB: this is the function taken from the VBA package (Daunizeau 2014), which does not use k-ToM's volatility parameter to avoid unidentifiability issues.
    """
    #Constants
    a = 0.36

    #Input variable transforms
    param_var = np.exp(param_var)

    #Prepare variance by weighing with gradient
    var_prepped = np.sum( (param_var * gradient**2), axis = 1)

    #Calculate
    p_opk = p_op_mean / np.sqrt(1 + a * var_prepped)

    #Output variable transform
    p_opk = inv_logit(p_opk)

    return p_opk

def expected_payoff_fun (p_op, agent, p_matrix):
    """
    Calculate expected payoff of choosing 1 over 0

    Examples:
    >>> 
    """
    #Calculate
    expected_payoff = (
        p_op * (p_matrix.outcome(1, 1, agent) - p_matrix.outcome(0, 1, agent)) + 
        (1 - p_op) * (p_matrix.outcome(1, 0, agent) - p_matrix.outcome(0, 0, agent)))

    return expected_payoff

def softmax (expected_payoff, b_temp): 
    """
    Softmax function for calculating own probability of choosing 1
    """
    #Input variable transforms
    b_temp = np.exp(b_temp)

    #Calculation
    p_self = 1 / (1 + np.exp(-(expected_payoff / b_temp)))

    #Set output bounds
    if p_self > 0.999:
        p_self = 0.999
        warn("Choice probability constrained at upper bound 0.999 to avoid rounding errors", Warning)
    if p_self < 0.001:
        p_self = 0.001
        warn("Choice probability constrained at lower bound 0.001 to avoid rounding errors", Warning)

    return p_self

def decision_function (
        new_internal_states,
        params,
        agent,
        level,
        p_matrix):
    """
    """
    #Extract needed parameters
    b_temp = params['behavioural_temperature']
    if 'bias' in params:
        bias = params['bias']
    #And variables
    p_op_mean0 = new_internal_states['own_states']['p_op_mean0']
    p_op_var0 = new_internal_states['own_states']['p_op_var0']
    p_op_mean = new_internal_states['own_states']['p_op_mean']
    param_var = new_internal_states['own_states']['param_var']
    gradient = new_internal_states['own_states']['gradient']
    p_k = new_internal_states['own_states']['p_k']

    #If (simulated) agent is a 0-ToM
    if level == 0:
        #Estimate probability of opponent choice
        p_op = p_op0_fun(p_op_mean0, p_op_var0)

    #If the (simulated) agent is a k-ToM
    else: 
        #Estimate probability of opponent choice for each simulated level
        p_opk = p_opk_fun(p_op_mean, param_var, gradient)

        #Weigh choice probabilities by level probabilities for aggregate choice probability estimate
        p_op = np.sum(p_opk * p_k)

    #Calculate expected payoff
    expected_payoff = expected_payoff_fun(p_op, agent, p_matrix)

    #Add bias
    if bias:
        expected_payoff = expected_payoff + bias

    #Softmax
    p_self = softmax(expected_payoff, b_temp)

    #Output variable transform
    p_self = logit(p_self)

    return p_self


#%% Learning functions
def p_op_var0_update (prev_p_op_mean0, prev_p_op_var0, volatility):
    """ 
    0-ToM updates variance / uncertainty on choice probability estimate

    Examples:
    >>> p_op_var0_update(1, 0.2, 1)
    0.8348496471878395
    
    >>> #Higher volatility results in a higher variance
    >>> p_op_var0_update(1, 0.2, 1) < p_op_var0_update (1, 0.2, 2) 
    True
    
    >>> #Mean close to 0.5 gives lower variance
    >>> p_op_var0_update(1, 0.45, 1) < p_op_var0_update (1, 0.2, 2)
    True
    """
    #Input variable transforms
    volatility = np.exp(volatility)
    prev_p_op_var0 = np.exp(prev_p_op_var0)
    prev_p_op_mean0 = inv_logit(prev_p_op_mean0)

    #Update
    new_p_op_var0 = ( 1 / (
       (1 / (volatility + prev_p_op_var0)) +
       prev_p_op_mean0 * (1 - prev_p_op_mean0)))

    #Output variable transform   
    new_p_op_var0 = np.log(new_p_op_var0)

    return new_p_op_var0


def p_op_mean0_update (prev_p_op_mean0, p_op_var0, op_choice):
    """
    0-ToM updates mean choice probability estimate
    """
    #Input variable transforms
    p_op_var0 = np.exp(p_op_var0)
    prev_p_op_mean0 = inv_logit(prev_p_op_mean0)
    
    #Update
    new_p_op_mean0 = prev_p_op_mean0 + p_op_var0 * (op_choice - prev_p_op_mean0)
    
    return new_p_op_mean0

def p_opk_approx_fun (prev_p_op_mean, prev_p_op_var, prev_gradient, level):
    """
    k-ToM
    Approximates the estimated choice probability of the opponent on the previous round. 
    A semi-analytical approximation derived in Daunizeau, J. (2017)
    """
    #Constants
    a = 0.205
    b = -0.319
    c = 0.781
    d = 0.870

    #Input variable transforms
    prev_p_op_var = np.exp(prev_p_op_var)

    #Prepare variance by weighing with gradient
    prev_var_prepped = []
    for level_index in range(level):
        prev_var_prepped[level_index] = prev_p_op_var[level,:].T.dot(prev_gradient[level,:]**2) 

    #Equation
    p_opk_approx = inv_logit (
        (prev_p_op_mean + b * prev_var_prepped^c) / np.sqrt(1 + a * prev_var_prepped^d))

    #Output variable transform
    p_opk_approx = np.log(p_opk_approx)

    return p_opk_approx

def p_k_udpate(prev_p_k, p_opk_approx, op_choice, dilution = None):
    """
    k-ToM updates its estimate of opponents sophistication level.
    If k-ToM has a dilution parameter, it does a partial forgetting of learned estimates.
    """
    
    #Input variable transforms
    p_opk_approx = np.exp(p_opk_approx)
    if dilution:
        dilution = inv_logit(dilution)
    
    #Do partial forgetting 
    if dilution:
        prev_p_k = (1 - dilution) * prev_p_k + dilution / len(prev_p_k)

    #Calculate
    new_p_k = (
        op_choice * 
        (prev_p_k * p_opk_approx / sum(prev_p_k * p_opk_approx)) +
        (1-op_choice) *
        (prev_p_k * (1-p_opk_approx) / sum(prev_p_k * (1- p_opk_approx))))

    #Force probability sum to 1
    if len(new_p_k) > 1:
        new_p_k[0] = 1 - sum(new_p_k[1:len(new_p_k)])

    return new_p_k

def param_var_update (prev_param_mean, prev_param_var, prev_gradient, p_k, volatility, volatility_dumy):
    """
    k-ToM updates its uncertainty / variance on its estimates of opponent's parameter values
    """
    #Input variable transforms
    prev_param_mean = inv_logit(prev_param_mean)
    prev_param_var = np.exp(prev_param_var)
    volatility = np.exp(volatility)

    #Calculate
    new_var = (
        1/
        (1 / (prev_param_var + volatility) +
        p_k * prev_param_mean * (1 - prev_param_mean) * prev_gradient**2))

    #Output variable transform
    new_var = np.log(new_var)

    return new_var

def param_mean_update (prev_p_op_mean, prev_param_mean, prev_gradient, p_k, param_var, op_choice):
    """
    k-ToM updates its estimates of opponent's parameter values
    """
    #Input variable transforms 
    param_var = np.exp(param_var)

    #Calculate
    new_param_mean = prev_param_mean + p_k * param_var * (op_choice - inv_logit(prev_param_mean))

    return new_param_mean

def gradient_update ():
    pass

def learning_function (
    prev_internal_states,
    params,
    self_choice,
    op_choice,
    level,
    agent,
    p_matrix
    ):

    #Extract needed parameters
    volatility = params['volatility']
    if 'dilution' in params:
        dilution = params['dilution']
    else:
        dilution = None
    #And variables
    prev_p_op_mean0 = prev_internal_states['own_states']['prev_p_op_mean0']
    prev_p_op_var0 = prev_internal_states['own_states']['p_op_var0']
    prev_p_op_mean = prev_internal_states['own_states']['prev_p_op_mean']
    prev_p_op_var = prev_internal_states['own_states']['prev_p_op_var']
    prev_gradient = prev_internal_states['own_states']['prev_gradient']
    prev_p_k = prev_internal_states['own_states']['prev_p_k']
    prev_gradient = prev_internal_states['own_states']['prev_gradient']

    #Make empty dictionary for storing updates states
    new_internal_states = {}

    #If the (simulated) agent is a 0-ToM
    if level == 0:
        #Update 0-ToM's uncertainty of opponent choice probability
        p_op_var0 = p_op_var0_update (prev_p_op_mean0, prev_p_op_var0, volatility)

        #Update 0-ToM's mean estimate of opponent choice probability
        p_op_mean0 = p_op_mean0_update (prev_p_op_mean0, p_op_var0, op_choice)

        #Gather own internal states 
        own_states = {'p_op_mean0':p_op_mean0, 'p_op_var0':p_op_var0}

    #If the (simulated) agent is a k-ToM
    else:
        #Update opponent level probabilities
        p_opk_approx = p_opk_approx_fun(prev_p_op_mean, prev_p_op_var, prev_gradient, level)
        p_k = p_k_udpate(prev_p_k, p_opk_approx, op_choice, dilution)

        #Update parameter estimates

    return new_internal_states



#%% Full Function

def k_tom (
    prev_internal_states,
    params,
    self_choice,
    op_choice,
    level,
    agent,
    p_matrix
    ):
    
    #Update estimates of opponent based on behaviour
    if self_choice:
        new_internal_states = learning_function(
        prev_internal_states,
        params,
        self_choice,
        op_choice,
        level,
        agent,
        p_matrix)

    else: #If first round or missed round, make no update
        new_internal_states = prev_internal_states

    #Calculate own decision probability
    p_self = decision_function(
        new_internal_states,
        params,
        agent,
        level,
        p_matrix)

    #Probability transform
    p_self = inv_logit(p_self)

    #Make decision
    choice = np.random.binomial(1, p_self)

    return (choice, new_internal_states)

#%% Other functions

def rec_prepare_k_tom ():
    pass

def logit (p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


#%% Testingfunctions
if __name__ == "__main__":
  import doctest
  doctest.testmod(verbose=True)