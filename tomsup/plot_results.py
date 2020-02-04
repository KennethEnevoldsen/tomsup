"""
Currently not in use. See plot.py for plottings functions

"""

# Plot functions
def plot_winnings(df, agent = 0):
    """
    assumes multiples simulations

    agent is either 1 or 0

    """
    import matplotlib.pyplot as plt

    cum_payoff = 'cum_payoff_a' + str(agent) 
    payoff_agent = 'payoff_agent' + str(agent)
    df[cum_payoff] = df.groupby(by = ['n_sim'])['payoff_agent0'].cumsum()

    plt.figure()
    # plot each line 
    for sim in range(df['n_sim'].max() + 1):
        tmp = df[['round', cum_payoff]].loc[df['n_sim'] == sim]
        plt.plot(tmp['round'], tmp[cum_payoff], color='grey', linewidth=1, alpha=0.2)

    # plot mean
    tmp = df.groupby(by = ['round'])[cum_payoff].mean()
    plt.plot(range(len(tmp)), tmp.values, color='lightblue', linewidth=4)
    plt.show()
    

def plot_actions(df, agent = 0):
    """
    assumes multiples simulations

    agent is either 1 or 0

    """
    import matplotlib.pyplot as plt

    action = 'action_agent' + str(agent)

    plt.figure()
    # plot each line 
    for sim in range(df['n_sim'].max() + 1):
        tmp = df[['round', action]].loc[df['n_sim'] == sim]
        plt.plot(tmp['round'], tmp[action], color='grey', linewidth=1, alpha=0.2)

    # plot mean
    tmp = df.groupby(by = ['round'])[action].mean()
    plt.plot(range(len(tmp)), tmp.values, color='lightblue', linewidth=4)
    plt.show()


def plot_own_states(agent, state = 'p_k'):
    """
    does not plot the first round

    agent is either 1 or 0


    """
    import matplotlib.pyplot as plt

    if state not in ['p_k', 'p_op_mean']:
        print("You can't plot the given state using this function.")
    df = agent.get_history()

    # plot mean
    tmp = df['internal_states'].apply(lambda d: d['own_states'][state][0,:])
    pk_df = pd.DataFrame(tmp.tolist())
    for col in pk_df:
        plt.plot(range(len(pk_df[col])), pk_df[col], label = col, linewidth=1)
    plt.legend()
    plt.show()


def plot_op_states(agent, op = 0, state = 'p_op_mean0'):
    """
    does not plot the first round

    agent is either 1 or 0


    """
    import matplotlib.pyplot as plt

    if state not in ['p_k', 'p_op_mean']:
        print("You can't plot the given state using this function.")
    df = agent.get_history()

    # plot mean
    tmp = df['internal_states'].apply(lambda d: d['opponent_states'][op]['own_states'][state])
    pk_df = pd.DataFrame(tmp.tolist())
    for col in pk_df:
        plt.plot(range(len(pk_df[col])), pk_df[col], label = col, linewidth=1)
    
    if len(pk_df.columns) > 1:
        plt.legend()
    plt.show()