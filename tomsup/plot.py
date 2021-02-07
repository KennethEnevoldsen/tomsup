"""
Utility plotting functiona for tomsup
"""
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class ResultsDf(pd.DataFrame):
    pass


def add_def_args(func, def_args):
    """
    func (fun): function which to add defaults arguments to
    def_args (dict): default argument given as a dictionary

    Examples
    >>> def addn(x,n):
    ...    return x + n
    >>> add3 = add_def_args(addn, {'n':3})
    >>> add3(2)
    5
    """

    def func_wrapper(*args, **kwargs):
        value = func(*args, **def_args, **kwargs)
        return value

    return func_wrapper


def mean_confidence_interval(x, confidence=0.95):
    return st.t.interval(confidence, len(x) - 1, loc=np.mean(x), scale=st.sem(x))


# aggregate_col='payoff_agent'
# aggregate_fun=np.mean
# certainty_fun='mean_ci_95'
# figsize=(11.7, 11.7)
# cmap="Blues"; dpi=300,
# na_color='xkcd:white'; x_axis=''; y_axis=''


def plot_heatmap(
    df,
    aggregate_col="payoff_agent",
    aggregate_fun=np.mean,
    certainty_fun="mean_ci_95",
    cmap="RdBu",
    na_color="xkcd:white",
    x_axis="",
    y_axis="",
):
    """
    df (ResultsDf): an outcome from the compete() function
    aggregate_fun (fun): Function which to aggregate by, defaults is mean
    certainty_fun (str | fun) function or string valid string include:
    mean_ci_X: where X is a float indicating the confidence.
    Default is mean_ci_95, i.e. a 95% confidence interval.
    """
    check_plot_input(df, None, None)
    df_ = df.copy()
    if isinstance(certainty_fun, str):
        if certainty_fun.startswith("mean_ci_"):
            ci = float(certainty_fun.split("_")[-1])
            certainty_fun = add_def_args(mean_confidence_interval, {"confidence": ci})

    # calc aggregate matrix
    df_mean = (
        df_[[aggregate_col + "0", "agent0", "agent1"]]
        .groupby(["agent0", "agent1"])
        .apply(lambda x: aggregate_fun(x))
        .reset_index()
    )
    df_mean2 = (
        df_[[aggregate_col + "1", "agent0", "agent1"]]
        .groupby(["agent0", "agent1"])
        .apply(lambda x: aggregate_fun(x))
        .reset_index()
    )
    df_mean.columns = ["agent0", "agent1", aggregate_col]
    df_mean2.columns = ["agent1", "agent0", aggregate_col]
    df_mean = pd.concat([df_mean, df_mean2])

    heat_df = pd.pivot(df_mean, values=aggregate_col, index="agent1", columns="agent0")

    if certainty_fun is not None:
        df_ci = (
            df_[[aggregate_col + "0", "agent0", "agent1"]]
            .groupby(["agent0", "agent1"])
            .apply(lambda x: mean_confidence_interval(x))
            .reset_index()
        )
        df_ci.columns = ["agent0", "agent1", "ci"]
        df_ci2 = (
            df_[[aggregate_col + "1", "agent0", "agent1"]]
            .groupby(["agent0", "agent1"])
            .apply(lambda x: mean_confidence_interval(x))
            .reset_index()
        )
        df_ci2.columns = ["agent1", "agent0", "ci"]
        df_ci = pd.concat([df_ci, df_ci2])

        df_ci["ci"] = [
            f"{round(m, 3)} \n({round(sd[0][0], 3)}, {round(sd[1][0], 3)})"
            for m, sd in zip(df_mean[aggregate_col], df_ci["ci"])
        ]
        annot_df = pd.pivot(df_ci, values="ci", index="agent1", columns="agent0")
        annot_df[annot_df.isna()] = ""

        fig, ax = plt.subplots(1, 1)
        p1 = sns.heatmap(heat_df, cmap=cmap, annot=annot_df.to_numpy(), fmt="")
        p1.set_facecolor(na_color)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
    else:
        fig, ax = plt.subplots(1, 1)
        p1 = sns.heatmap(heat_df, cmap=cmap, fmt="")
        p1.set_facecolor(na_color)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
    plt.show()


def check_plot_input(df, agent0, agent1):
    if not isinstance(df, ResultsDf):
        raise ValueError(
            "The input dataframe is expected to be a ResultDf \
                          which it is not. ResultsDf is a subclass of pandas \
                          DataFrame which is obtained using the compete() \
                          function"
        )
    if (agent0 not in df["agent0"].values) and not (agent0 is None):
        raise ValueError(
            "The specified agent0 is not a valid agent \
                          (i.e. it is not present in columns agent0)"
        )
    if (agent1 not in df["agent1"].values) and not (agent1 is None):
        raise ValueError(
            "The specified agent1 is not a valid agent \
                          (i.e. it is not present in columns agent1)"
        )


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
    df = df.loc[(df["agent0"] == agent0) & (df["agent1"] == agent1)].copy()

    cum_payoff = "cum_payoff_a" + str(agent)
    df[cum_payoff] = df.groupby(by=["n_sim"])["payoff_agent0"].cumsum()

    plt.figure()
    if "n_sim" in df:
        # plot each line for each sim
        for sim in range(df["n_sim"].max() + 1):
            tmp = df[["round", cum_payoff]].loc[df["n_sim"] == sim]
            plt.plot(
                tmp["round"], tmp[cum_payoff], color="grey", linewidth=1, alpha=0.2
            )

    # plot mean
    # set label text
    label_text = "mean score across simulations" if "n_sim" in df else "score"

    tmp = df.groupby(by=["round"])[cum_payoff].mean()
    plt.plot(
        range(len(tmp)), tmp.values, color="lightblue", linewidth=4, label=label_text
    )
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.show()


def choice(df, agent0, agent1, agent=0, plot_individual_sim=False):
    """
    assumes multiples simulations

    agent is either 1 or 0
    """
    check_plot_input(df, agent0, agent1)

    plt.clf()
    df = df.loc[(df["agent0"] == agent0) & (df["agent1"] == agent1)].copy()

    action = "choice_agent" + str(agent)

    plt.figure()
    # plot each line
    if plot_individual_sim:
        for sim in range(df["n_sim"].max() + 1):
            tmp = df[["round", action]].loc[df["n_sim"] == sim]
            plt.plot(tmp["round"], tmp[action], color="grey", linewidth=1, alpha=0.2)

    label_text = "mean score across simulations" if "n_sim" in df else "score"

    # plot mean
    tmp = df.groupby(by=["round"])[action].mean()
    plt.plot(
        range(len(tmp)), tmp.values, color="lightblue", linewidth=4, label=label_text
    )
    plt.legend(loc="upper right")
    plt.xlabel("Round")
    plt.ylabel("Choice")
    plt.ylim(0, 1)
    plt.show()


def plot_history(
    df, agent0, agent1, state, agent=0, fun=lambda x: x[state], ylab="", xlab="Round"
):
    """
    df (ResultsDf): an outcome from the compete() function
    agent0 (str): an agent name in the agent0 column in the df
    agent1 (str): an agent name in the agent1 column in the df
    agent (0|1): An indicate of which agent of agent 0 and 1 you wish to plot
    state (str): a state of the agent you wish to plot.
    """
    check_plot_input(df, agent0, agent1)

    plt.clf()
    df = df.loc[(df["agent0"] == agent0) & (df["agent1"] == agent1)].copy()

    hist = "history_agent" + str(agent)
    plt.figure()
    # plot each line
    for sim in range(df["n_sim"].max() + 1):
        tmp = df[["round", hist]].loc[df["n_sim"] == sim]
        tmp[hist].apply(fun)
        plt.plot(
            tmp["round"],
            tmp[hist].apply(fun).values,
            color="grey",
            linewidth=1,
            alpha=0.2,
        )
    # plot mean
    label_text = "mean score across simulations" if "n_sim" in df else "score"
    df["extract"] = df[hist].apply(fun)
    tmp = df.groupby(by=["round"])["extract"].mean()
    plt.plot(
        range(len(tmp)), tmp.values, color="lightblue", linewidth=4, label=label_text
    )
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def plot_p_k(df, agent0, agent1, level, agent=0):
    plot_history(
        df,
        agent0,
        agent1,
        state="p_k",
        agent=agent,
        fun=lambda x: x["internal_states"]["own_states"]["p_k"][level],
        ylab=f"Probability of k={level}",
        xlab="Round",
    )


def plot_p_op_1(df, agent0, agent1, agent=0):
    """"""
    plot_history(
        df,
        agent0,
        agent1,
        state="p_op",
        agent=agent,
        fun=lambda x: x["internal_states"]["own_states"]["p_op"][0],
    )


def plot_p_self(df, agent0, agent1, agent=0):
    """"""
    plot_history(
        df,
        agent0,
        agent1,
        state="p_self",
        agent=agent,
        fun=lambda x: x["internal_states"]["own_states"]["p_self"],
    )


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
    plot_history(
        df,
        agent0,
        agent1,
        state="p_op",
        agent=agent,
        fun=lambda x: x["internal_states"]["opponent_states"][level]["own_states"][
            state
        ],
    )
