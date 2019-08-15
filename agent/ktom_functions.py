#%% Decision functions


def basic_p_op_1_fun ():
    pass

def p_op_1_k_fun():
    pass

#Calculate expected payoff of choosing 1 over 0
def expected_payoff_difference (p_op1, agent, p_matrix):
    return (p_op1 * (p_matrix.outcome(1, 1, agent) - p_matrix.outcome(0, 1, agent)) + 
           (1 - p_op1) * (p_matrix.outcome(1, 0, agent) - p_matrix.outcome(0, 0, agent)))

#Softmax function for calculating own probability of choosing 1
def softmax (e_pay_diff, b_temp): 
    return 1 / (1 + exp(-(e_pay_diff / b_temp)))
    #SKAL DER UPPER OG LOWER BOUNDS?

def decision_function ()





#%% Learning functions

def basic_variance_update ()

def basic_mean_update () 

def p_op_1_k_approx_fun ()

def udpate_p_k()

def parameter_variance_update ()

def parameter_mean_update ()

def gradient_update ()

def rec_learning_function ()



#%% Other functions

def k_tom ()

def rec_prepare_k_tom ()
