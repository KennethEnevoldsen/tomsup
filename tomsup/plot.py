"""
Utility plotting functiona for tomsup
"""
from typing import Callable, Optional, Union
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from functools import partial


class ResultsDf(pd.DataFrame):
    """A class wrapper around a pandas dataframe for denoting results from a compete function.
    Function exactly like a pandas dataframe.
    """


def mean_confidence_interval(x: np.array, confidence: float = 0.95) -> np.array:
    return st.t.interval(confidence, len(x) - 1, loc=np.mean(x), scale=st.sem(x))


def plot_heatmap(
    df: pd.DataFrame,
    aggregate_col: str = "payoff_agent",
    aggregate_fun: Callable = np.mean,
    certainty_fun: Union[Callable, str] = "mean_ci_95",
    cmap: str = "RdBu",
    na_color: str = "xkcd:white",
    xlab: str = "",
    ylab: str = "",
    cbarlabel: str = "Average score of the agent",
    show: bool = True,
) -> None:
    """plot a heatmap of the agents payoffs

    Args:
        df (pd.DataFrame): An outcome from the compete() function
        aggregate_col (str, optional): Column to be aggregated pr agent. Defaults to "payoff_agent".
        aggregate_fun (Callable, optional): Function which to aggregate by, defaults is mean. Defaults to np.mean.
        certainty_fun (Union[Callable, str], optional): function should estimate uncertainty or string. Valid string include, mean_ci_X:
            where X is a float indicating the confidence interval. Defaults to "mean_ci_95".
        cmap (str, optional): The color map. Defaults to "RdBu".
        na_color (str, optional): The nan color. Defaults to "xkcd:white", e.g. white.
        xlab (str, optional): The name on the x axis. Defaults to "".
        ylab (str, optional): The name on the y axis. Defaults to "".
        charlabel (str, optional): The label on the color bar, defaults to "Average score of the Agent."
        show (bool, optional): Should plt.show be run at the end. Defaults to True.
    """
    check_plot_input(df, None, None)
    df_ = df.copy()
    if isinstance(certainty_fun, str):
        if certainty_fun.startswith("mean_ci_"):
            ci = float(certainty_fun.split("_")[-1])
            certainty_fun = partial(mean_confidence_interval, confidence=ci)

    # calc aggregate matrix
    df_mean = (
        df_[[aggregate_col + "0", "agent0", "agent1"]]
        .groupby(["agent0", "agent1"])
        .apply(aggregate_fun)
        .reset_index()
    )
    df_mean2 = (
        df_[[aggregate_col + "1", "agent0", "agent1"]]
        .groupby(["agent0", "agent1"])
        .apply(aggregate_fun)
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
            .apply(mean_confidence_interval)
            .reset_index()
        )
        df_ci.columns = ["agent0", "agent1", "ci"]
        df_ci2 = (
            df_[[aggregate_col + "1", "agent0", "agent1"]]
            .groupby(["agent0", "agent1"])
            .apply(mean_confidence_interval)
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

        ax = sns.heatmap(
            heat_df,
            cmap=cmap,
            annot=annot_df.to_numpy(),
            fmt="",
            cbar_kws={"label": cbarlabel},
        )
        ax.set_facecolor(na_color)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    else:
        ax = sns.heatmap(heat_df, cmap=cmap, fmt="")
        ax.set_facecolor(na_color)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    if show is True:
        plt.show()


def check_plot_input(df: pd.DataFrame, agent0: str, agent1: str) -> None:
    """checks if plot input is valid"""
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


def score(
    df: pd.DataFrame, agent0: str, agent1: str, agent: int = 0, show: bool = True
):
    """plot the score of the agent pair

    Args:
        df (pd.DataFrame): a dataframe resulting from a compete function on an
            AgentGroup
        agent0 (str): agent0 in the agent pair which you seek to plot, by default
            it plot agent0 performance vs. agent1, to plot agent1 set agent = 1.
        agent1 (str): agent1 in the agent pair which you seek to plot
        agent (int, optional): Indicate whether you should plot the score of agent 0 or 1. Defaults to 0.
        show (bool, optional): Should plt.show be run at the end. Defaults to True.

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
        >>> penny = ts.PayoffMatrix("penny_competive")
        >>> results = group.compete(p_matrix = penny, n_rounds = 20, n_sim = 4)
        >>> ts.plot.score(results, agent0 = "RB", agent1 = "QL", agent = 0)
    """
    check_plot_input(df, agent0, agent1)

    plt.clf()
    df = df.loc[(df["agent0"] == agent0) & (df["agent1"] == agent1)].copy()

    cum_payoff = "cum_payoff_a" + str(agent)
    df[cum_payoff] = df.groupby(by=["n_sim"])["payoff_agent" + str(agent)].cumsum()

    fig, ax = plt.subplots(1, 1)

    if "n_sim" in df:
        # plot each line for each sim
        for sim in range(df["n_sim"].max() + 1):
            tmp = df[["round", cum_payoff]].loc[df["n_sim"] == sim]
            ax.plot(tmp["round"], tmp[cum_payoff], color="grey", linewidth=1, alpha=0.2)

    # plot mean
    # set label text
    label_text = "mean score across simulations" if "n_sim" in df else "score"

    tmp = df.groupby(by=["round"])[cum_payoff].mean()
    ax.plot(
        range(len(tmp)), tmp.values, color="lightblue", linewidth=4, label=label_text
    )
    ax.legend()
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    a_name = agent1 if agent == 1 else agent0
    op_name = agent1 if agent != 1 else agent0
    ax.set_title(f"{a_name} playing against {op_name}")
    if show is True:
        plt.show()


def choice(
    df: pd.DataFrame,
    agent0: str,
    agent1: str,
    agent: int = 0,
    sim: Optional[int] = None,
    plot_individual_sim: bool = False,
    show: bool = True,
):
    """plot the score of the agent pair

    Args:
        df (pd.DataFrame): a dataframe resulting from a compete function on an
            AgentGroup
        agent0 (str): agent0 in the agent pair which you seek to plot, by default
            it plot agent0 performance vs. agent1, to plot agent1 set agent = 1.
        agent1 (str): agent1 in the agent pair which you seek to plot
        agent (int, optional): Indicate whether you should plot the choice of agent 0 or 1. Defaults to 0.
        sim: (Optional[int], optional): A specific simulation you wish to plot. Defualts to None
            indicating it should plot all simulations.
        plot_individual_sim (bool, optional): Should individual simulations be plotted. Defaults to false.
        show (bool, optional): Should plt.show be run at the end. Defaults to True.
    """
    check_plot_input(df, agent0, agent1)

    plt.clf()
    df = df.loc[(df["agent0"] == agent0) & (df["agent1"] == agent1)].copy()
    if sim is not None:
        df = df.loc[df["n_sim"] == sim]
        df = df.drop(columns=["n_sim"])

    action = "choice_agent" + str(agent)

    fig, ax = plt.subplots(1, 1)
    # plot each line
    if plot_individual_sim:
        for sim in range(df["n_sim"].max() + 1):
            tmp = df[["round", action]].loc[df["n_sim"] == sim]
            ax.plot(tmp["round"], tmp[action], color="grey", linewidth=1, alpha=0.2)

    label_text = "mean choice across simulations" if "n_sim" in df else "choice"

    # plot mean
    tmp = df.groupby(by=["round"])[action].mean()
    ax.plot(
        range(len(tmp)), tmp.values, color="lightblue", linewidth=4, label=label_text
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("Round")
    ax.set_ylabel("Choice")
    a_name = agent1 if agent == 1 else agent0
    op_name = agent1 if agent != 1 else agent0
    ax.set_title(f"{a_name} playing against {op_name}")

    ax.set_ylim(0, 1)
    if show is True:
        plt.show()


def plot_history(
    df: pd.DataFrame,
    agent0: str,
    agent1: str,
    state: str,
    agent: int = 0,
    fun: Callable = lambda x: x[state],
    ylab: str = "",
    xlab: str = "Round",
    show: bool = True,
) -> None:
    """plot the history of an agent.

    Args:
        df (pd.DataFrame): a dataframe resulting from a compete function on an
            AgentGroup
        agent0 (str): agent0 in the agent pair which you seek to plot, by default
            it plot agent0 performance vs. agent1, to plot agent1 set agent = 1.
        agent1 (str): agent1 in the agent pair which you seek to plot
        state (str): The state of the agent you wish to plot.
        agent (int, optional): Indicate whether you should plot the score of agent 0 or 1. Defaults to 0.
        fun (Callable, optional): A getter function for extracting the state. Defaults to lambdax:x[state].
        ylab (str, optional): Label on y-axis. Defaults to "".
        xlab (str, optional): Label on the x-axis. Defaults to "Round".
        show (bool, optional): Should plt.show be run at the end. Defaults to True.
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
    a_name = agent1 if agent == 1 else agent0
    op_name = agent1 if agent != 1 else agent0
    plt.title(f"{a_name} playing against {op_name}")

    if show is True:
        plt.show()


def plot_p_k(
    df: pd.DataFrame, agent0: str, agent1: str, level: int, agent=0, show: bool = True
) -> None:
    """plot the p_k of a k-ToM agent

    Args:
        df (pd.DataFrame): a dataframe resulting from a compete function on an
            AgentGroup
        agent0 (str): agent0 in the agent pair which you seek to plot, by default
            it plot agent0 performance vs. agent1, to plot agent1 set agent = 1.
        agent1 (str): agent1 in the agent pair which you seek to plot
        level (int): The sophistication level to plot
        agent (int, optional): Indicate whether you should plot the score of agent 0 or 1. Defaults to 0.
        show (bool, optional): Should plt.show be run at the end. Defaults to True.
    """
    plot_history(
        df,
        agent0,
        agent1,
        state="p_k",
        agent=agent,
        fun=lambda x: x["internal_states"]["own_states"]["p_k"][level],
        ylab=f"Probability of k={level}",
        xlab="Round",
        show=show,
    )


def plot_p_op_1(
    df: pd.DataFrame, agent0: str, agent1: str, agent: int = 0, show: bool = True
) -> None:
    """plot the p_op_1 of a k-ToM agent

    Args:
        df (pd.DataFrame): a dataframe resulting from a compete function on an
            AgentGroup
        agent0 (str): agent0 in the agent pair which you seek to plot, by default
            it plot agent0 performance vs. agent1, to plot agent1 set agent = 1.
        agent1 (str): agent1 in the agent pair which you seek to plot
        agent (int, optional): Indicate whether you should plot the score of agent 0 or 1. Defaults to 0.
        show (bool, optional): Should plt.show be run at the end. Defaults to True.
    """
    plot_history(
        df,
        agent0,
        agent1,
        state="p_op",
        agent=agent,
        fun=lambda x: x["internal_states"]["own_states"]["p_op"][0],
        show=show,
    )


def plot_p_self(
    df: pd.DataFrame, agent0: str, agent1: str, agent: int = 0, show: bool = True
) -> None:
    """plot the p_self of a k-ToM agent

    Args:
        df (pd.DataFrame): a dataframe resulting from a compete function on an
            AgentGroup
        agent0 (str): agent0 in the agent pair which you seek to plot, by default
            it plot agent0 performance vs. agent1, to plot agent1 set agent = 1.
        agent1 (str): agent1 in the agent pair which you seek to plot
        agent (int, optional): Indicate whether you should plot the score of agent 0 or 1. Defaults to 0.
        show (bool, optional): Should plt.show be run at the end. Defaults to True.
    """
    plot_history(
        df,
        agent0,
        agent1,
        state="p_self",
        agent=agent,
        fun=lambda x: x["internal_states"]["own_states"]["p_self"],
        show=show,
    )


def plot_op_states(
    df: pd.DataFrame,
    agent0: str,
    agent1: str,
    state: str,
    level: int = 0,
    agent: int = 0,
    show: bool = True,
):
    """
    df (ResultsDf): an outcome from the compete() function
    agent0 (str): an agent name in the agent0 column in the df
    agent1 (str): an agent name in the agent1 column in the df
    agent (0|1): An indicate of which agent of agent 0 and 1 you wish to plot
    the indicated agent must be a theory of mind agent (ToM)
    state (str): a state of the simulated opponent you wish to plot.
    level (str): level of the similated opponent you wish to plot.
    show (bool, optional): Should plt.show be run at the end. Defaults to True.
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
        show=show,
    )
