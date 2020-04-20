"""
Plot function for tomsup
"""
import matplotlib.pyplot as plt
from tomsup.agent import ResultsDf


def check_plot_input(df, agent0, agent1):
    if not isinstance(df, ResultsDf):
        raise ValueError("The input dataframe is expected to be a ResultDf \
                          which it is not. ResultsDf is a subclass of pandas \
                          DataFrame which is obtained using the compete() \
                          function")
    if agent0 not in df['agent0'].values:
        raise ValueError("The specified agent0 is not a valid agent \
                          (i.e. it is not present in columns agent0)")
    if agent1 not in df['agent1'].values:
        raise ValueError("The specified agent1 is not a valid agent \
                          (i.e. it is not present in columns agent1)")


def score(df, agent0, agent1, agent=0):
    """
    df (pd.DataFrame): a dataframe resulting from a compete function on an
    AgentGroup
    agent0 (str): agent0 in the agent pair which you seek to plot, by default
    it plot agent0 performance vs. agent1, to plot agent1 set agent = 1.
    agent1 (str): agent1 in the agent pair which you seek to plot
    agent (0 | 1): Indicate whether you should plot the score of agent 0 or 1.

    Examples:
    >>> agents = ['RB', 'QL', 'WSLS'] # create a list of agents
    >>> start_params = [{'bias': 0.7}, {'learning_rate': 0.5}, {}]
    >>> # create a list of their starting parameters
    >>> # (an empty dictionary {} simply assumes defaults)
    >>> # create a group of agents
    >>> group = ts.create_agents(agents, start_params)
    >>> # round_robin e.g. each agent will play against all other agents
    >>> group.set_env(env = 'round_robin')
    >>> # make them compete for 4 simulations
    >>> results = group.compete(p_matrix = penny, n_rounds = 20, n_sim = 4)
    >>> ts.plot.score(results, agent0 = "RB", agent1 = "QL", agent = 0)
    """
    check_plot_input(df, agent0, agent1)

    plt.clf()
    df = df.loc[(df['agent0'] == agent0) & (df['agent1'] == agent1)].copy()

    cum_payoff = 'cum_payoff_a' + str(agent)
    df[cum_payoff] = df.groupby(by=['n_sim'])['payoff_agent0'].cumsum()

    plt.figure()
    if 'n_sim' in df:
        # plot each line for each sim
        for sim in range(df['n_sim'].max() + 1):
            tmp = df[['round', cum_payoff]].loc[df['n_sim'] == sim]
            plt.plot(tmp['round'], tmp[cum_payoff], color='grey', linewidth=1,
                     alpha=0.2)

    # plot mean
        # set label text
    label_text = 'mean score across simulations' if 'n_sim' in df else 'score'

    tmp = df.groupby(by=['round'])[cum_payoff].mean()
    plt.plot(range(len(tmp)), tmp.values, color='lightblue', linewidth=4,
             label=label_text)
    plt.legend()
    plt.show()


def choice(df, agent0, agent1, agent=0):
    """
    assumes multiples simulations

    agent is either 1 or 0
    """
    check_plot_input(df, agent0, agent1)

    plt.clf()
    df = df.loc[(df['agent0'] == agent0) & (df['agent1'] == agent1)].copy()

    action = 'choice_agent' + str(agent)

    plt.figure()
    # plot each line
    for sim in range(df['n_sim'].max() + 1):
        tmp = df[['round', action]].loc[df['n_sim'] == sim]
        plt.plot(tmp['round'], tmp[action], color='grey', linewidth=1,
                 alpha=0.2)

    # plot mean
    tmp = df.groupby(by=['round'])[action].mean()
    plt.plot(range(len(tmp)), tmp.values, color='lightblue', linewidth=4)
    plt.show()


def plot_history(df, agent0, agent1, state, agent=0, fun=lambda x: x[state]):
    """
    df (ResultsDf): an outcome from the compete() function
    agent0 (str): an agent name in the agent0 column in the df
    agent1 (str): an agent name in the agent1 column in the df
    agent (0|1): An indicate of which agent of agent 0 and 1 you wish to plot
    state (str): a state of the agent you wish to plot.
    """
    check_plot_input(df, agent0, agent1)

    plt.clf()
    df = df.loc[(df['agent0'] == agent0) & (df['agent1'] == agent1)].copy()

    hist = 'history_agent' + str(agent)

    plt.figure()
    # plot each line
    for sim in range(df['n_sim'].max() + 1):
        tmp = df[['round', hist]].loc[df['n_sim'] == sim]
        tmp[hist].apply(fun)
        plt.plot(tmp['round'], tmp[hist].apply(fun).values,
                 color='grey', linewidth=1,
                 alpha=0.2)
    # plot mean
    df['extract'] = df[hist].apply(fun)
    tmp = df.groupby(by=['round'])['extract'].mean()
    plt.plot(range(len(tmp)), tmp.values, color='lightblue', linewidth=4)
    plt.show()


def plot_p_k(df, agent0, agent1, agent=0):
    plot_history(df, agent0, agent1, state="p_k", agent=agent,
                 fun=lambda x: x['own_states'][state][0])


def plot_p_op_1(df, agent0, agent1, agent=0):
    plot_history(df, agent0, agent1, state="p_op", agent=agent,
                 fun=lambda x: x['own_states'][state][0])


def plot_p_self(df, agent0, agent1, agent=0):
    plot_history(df, agent0, agent1, state="p_self", agent=agent,
                 fun=lambda x: x['own_states'][state])


def plot_op_states(df, agent0, agent1, state, level=0, agent=0):
    """
    df (ResultsDf): an outcome from the compete() function
    agent0 (str): an agent name in the agent0 column in the df
    agent1 (str): an agent name in the agent1 column in the df
    agent (0|1): An indicate of which agent of agent 0 and 1 you wish to plot
    the indicated agent must be a theory of mind agent (ToM)
    state (str): a state of the simulated opponent you wish to plot.
    level (str): level of the similated opponent you wish to plot.
    """
    plot_history(df, agent0, agent1, state="p_op", agent=agent,
                 fun=lambda x: x['internal_states'][level]['opponent_states'][state])

