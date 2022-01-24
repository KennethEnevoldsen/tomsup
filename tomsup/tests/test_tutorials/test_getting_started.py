import tomsup as ts


def test_tutorial():
    jung = ts.RB(
        bias=0.7, save_history=True
    )  # calling the agent subclass RB - for more on save_history see '3) inspecting Agent and AgentGroup'

    # Let's examine the jung
    print(f"jung is an class of type: {type(jung)}")
    if isinstance(jung, ts.Agent):
        print(f"but jung is also an instance of the parent class ts.Agent")

    # let us have Jung make a choice
    choice = jung.compete()

    print(
        f"jung chose {choice} and his probability for choosing 1 was {jung.get_bias()}."
    )

    skinner = ts.create_agents(
        agents="QL", start_params={"save_history": True}
    )  # create a reinforcement learning agent

    penny = ts.PayoffMatrix(
        name="penny_competitive"
    )  # fetch the competitive matching pennies game.

    # print the payoff matrix
    print(penny)

    # fetch the underlying numpy matrix
    print(penny.get_matrix())

    jung_a = jung.compete()  # a for action
    skinner_a = skinner.compete(
        p_matrix=penny, agent=1, op_choice=None
    )  # Note that op_choice can be unspecified (or None) in the first round

    jung_p = penny.payoff(choice_agent0=jung_a, choice_agent1=skinner_a, agent=0)
    skinner_p = penny.payoff(choice_agent0=jung_a, choice_agent1=skinner_a, agent=1)

    print(
        f"jung chose {jung_a} and skinner chose {skinner_a}, which results in a payoff for jung of {jung_p} and skinner of {skinner_p}."
    )
    # Note that you might get different results simply by chance

    results = ts.compete(
        jung, skinner, p_matrix=penny, n_rounds=4, save_history=True, verbose=True
    )
    print(type(results))

    jung_sum = results["payoff_agent0"].sum()
    skinner_sum = results["payoff_agent1"].sum()

    print(
        f"jung seemed to get a total of {jung_sum} points, while skinner got a total of {skinner_sum}."
    )

    results.head()  # inspect the first 5 rows of the df

    results = ts.compete(
        jung,
        skinner,
        penny,
        n_rounds=4,
        n_sim=2,
        save_history=True,
        return_val="df",
        verbose=False,
    )
    results.head()

    agents = ["RB", "QL", "WSLS"]  # create a list of agents
    start_params = [
        {"bias": 0.7},
        {"learning_rate": 0.5},
        {},
    ]  # create a list of their starting parameters (an empty dictionary {} simply assumes defaults)

    group = ts.create_agents(agents, start_params)  # create a group of agents
    print(group)
    print("\n----\n")  # to space out the outputs

    group.set_env(
        env="round_robin"
    )  # round_robin e.g. each agent will play against all other agents

    # make them compete
    group.compete(p_matrix=penny, n_rounds=4, n_sim=2, verbose=True)
    results = group.get_results()
    results.head()  # examine the first 5 rows in results

    # What if I want to know the starting parameters?
    print(
        "This is the starting parameters of jung: ", jung.get_start_params()
    )  # Note that it also prints out default parameters
    print("This is the starting parameters of skinner: ", skinner.get_start_params())

    # What if I want to know the agent last choice?
    print("This is jung's last choice: ", jung.get_choice())
    print("This is skinner's last choice: ", skinner.get_choice())

    # What if I want to know the agents strategy?
    print("jung's strategy is: ", jung.get_strategy())
    print("skinner's strategy is: ", skinner.get_strategy())

    # What is the history of skinner (e.g. what is his choices and internal states)

    history = jung.get_history(format="df")
    print(history.head())

    print("\n --- \n")  # for spacing

    history = skinner.get_history(format="df")
    print(history.head(15))  # the first 15 rows

    ts.plot.score(results, agent0="RB", agent1="QL", agent=0, show=False)

    ts.plot.choice(results, agent0="RB", agent1="QL", agent=0, show=False)

    # Create a list of agents
    agents = ["RB", "QL", "WSLS", "1-TOM", "2-TOM"]
    # And set their starting parameters. An empty dict denotes default values
    start_params = [{"bias": 0.7}, {"learning_rate": 0.5}, {}, {}, {}]

    group = ts.create_agents(agents, start_params)  # create a group of agents

    # Specify the environment
    # round_robin e.g. each agent will play against all other agents
    group.set_env(env="round_robin")

    # Finally, we make the group compete 20 simulations of 30 rounds
    group.compete(p_matrix=penny, n_rounds=4, n_sim=2, save_history=True)

    res = group.get_results()
    res.head(1)  # print the first row

    import matplotlib.pyplot as plt

    # Set figure size
    plt.rcParams["figure.figsize"] = [10, 10]

    group.plot_heatmap(cmap="RdBu", show=False)

    group.plot_choice(
        agent0="RB", agent1="QL", agent=0, plot_individual_sim=False, show=False
    )

    group.plot_score(agent0="RB", agent1="QL", agent=0, show=False)

    group.plot_p_k(agent0="1-TOM", agent1="2-TOM", agent=1, level=0, show=False)
    group.plot_p_k(agent0="1-TOM", agent1="2-TOM", agent=1, level=1, show=False)

    group.plot_history(
        "1-TOM",
        "2-TOM",
        agent=1,
        state="",
        fun=lambda x: x["internal_states"]["own_states"]["p_op_mean"][0],
        show=False,
    )

    df = group.get_results()
    print(
        df.loc[(df["agent0"] == "1-TOM") & (df["agent1"] == "2-TOM")]["history_agent1"][
            1
        ]["internal_states"]["opponent_states"][1]["own_states"]
    )

    # volatility
    group.plot_history(
        "1-TOM",
        "2-TOM",
        agent=1,
        state="",
        fun=lambda x: x["internal_states"]["opponent_states"][1]["own_states"][
            "param_mean"
        ][0, 0],
        ylab="Volalitity (log-odds)",
        show=False,
    )
    #

    # behav temp
    group.plot_history(
        "1-TOM",
        "2-TOM",
        agent=1,
        state="",
        fun=lambda x: x["internal_states"]["opponent_states"][1]["own_states"][
            "param_mean"
        ][0, 1],
        ylab="Behavioral Temperature (log-odds)",
        show=False,
    )

    # ktom simple example
    tom_1 = ts.TOM(level=1, dilution=None, save_history=True)

    # Extact the parameters
    print(tom_1.get_parameters())

    tom_2 = ts.TOM(
        level=2,
        volatility=-2,
        b_temp=-2,  # more deterministic
        bias=0,
        dilution=None,
        save_history=True,
    )

    choice = tom_2.compete(p_matrix=penny, agent=0, op_choice=None)
    print(choice)

    tom_2.reset()  # reset before start

    prev_choice_1tom = None
    prev_choice_2tom = None
    for trial in range(1, 4):
        # note that op_choice is choice on previous turn
        # and that agent is the agent you repond to the in payoff matrix
        choice_1 = tom_1.compete(p_matrix=penny, agent=0, op_choice=prev_choice_1tom)
        choice_2 = tom_2.compete(p_matrix=penny, agent=1, op_choice=prev_choice_2tom)

        # update previous choice
        prev_choice_1tom = choice_1
        prev_choice_2tom = choice_2

        print(
            f"Round {trial}",
            f"  1-ToM choose {choice_1}",
            f"  2-ToM choose {choice_2}",
            sep="\n",
        )

    tom_2.print_internal(keys=["p_k", "p_op"], level=[0, 1])
