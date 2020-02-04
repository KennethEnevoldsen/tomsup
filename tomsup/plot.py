"""
Plot function for tomsup 
"""

def score(df, agent0, agent1, agent = 0):
    """
    df (pd.DataFrame): a dataframe resulting from a compete function on an AgentGroup
    agent0 (str): agent0 in the agent pair which you seek to plot, by default it plot agent0 performance vs. agent1, to plot agent1 set agent = 1.
    agent1 (str): agent1 in the agent pair which you seek to plot
    agent (0 | 1): Indicate whether you should plot the score of agent 0 or 1.

    Examples:
    >>> agents = ['RB', 'QL', 'WSLS'] # create a list of agents
    >>> start_params = [{'bias': 0.7}, {'learning_rate': 0.5}, {}] # create a list of their starting parameters (an empty dictionary {} simply assumes defaults)
    >>> group = ts.create_agents(agents, start_params) # create a group of agents
    >>> group.set_env(env = 'round_robin') # round_robin e.g. each agent will play against all other agents
    >>> results = group.compete(p_matrix = penny, n_rounds = 20, n_sim = 4) # make them compete for 4 simulations
    >>> ts.plot score(results, agent0 = "RB", agent1 = "QL", agent = 0)
    """
    import matplotlib.pyplot as plt

    plt.clf()
    df = df.loc[(df['agent0'] == agent0) & (df['agent1'] == agent1)].copy()

    cum_payoff = 'cum_payoff_a' + str(agent) 
    payoff_agent = 'payoff_agent' + str(agent)
    df[cum_payoff] = df.groupby(by = ['n_sim'])['payoff_agent0'].cumsum()

    plt.figure()
    if 'n_sim' in df:
        # plot each line for each sim
        for sim in range(df['n_sim'].max() + 1):
            tmp = df[['round', cum_payoff]].loc[df['n_sim'] == sim]
            plt.plot(tmp['round'], tmp[cum_payoff], color='grey', linewidth=1, alpha=0.2)

    # plot mean
        # set label text
    label_text = 'mean score across simulations' if 'n_sim' in df else 'score'

    tmp = df.groupby(by = ['round'])[cum_payoff].mean()
    plt.plot(range(len(tmp)), tmp.values, color='lightblue', linewidth=4, label=label_text)
    plt.legend()
    plt.show()


def choice(df,  agent0 = "RB", agent1 = "QL", agent = 0):
    """
    assumes multiples simulations

    agent is either 1 or 0

    """
    import matplotlib.pyplot as plt

    plt.clf()
    df = df.loc[(df['agent0'] == agent0) & (df['agent1'] == agent1)].copy()

    action = 'choice_agent' + str(agent)

    plt.figure()
    # plot each line 
    for sim in range(df['n_sim'].max() + 1):
        tmp = df[['round', action]].loc[df['n_sim'] == sim]
        plt.plot(tmp['round'], tmp[action], color='grey', linewidth=1, alpha=0.2)

    # plot mean
    tmp = df.groupby(by = ['round'])[action].mean()
    plt.plot(range(len(tmp)), tmp.values, color='lightblue', linewidth=4)
    plt.show()
