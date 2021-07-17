import random

import tomsup as ts


def test_tutorial():
    random.seed(1995)

    # initiate the competitive matching pennies game
    penny = ts.PayoffMatrix(name="penny_competitive")

    # print the payoff matrix
    print(penny)

    # define the random bias agent, which chooses 1 70 percent of the time, and call the agent "jung"
    jung = ts.RB(bias=0.7)

    # Examine Agent
    print(f"jung is a class of type: {type(jung)}")
    if isinstance(jung, ts.Agent):
        print(f"but jung is also an instance of the parent class ts.Agent")

    # let us have Jung make a choice
    choice = jung.compete()

    print(
        f"jung chose {choice} and his probability for choosing 1 was {jung.get_bias()}."
    )

    # create a reinforcement learning agent
    skinner = ts.create_agents(agents="QL", start_params={"save_history": True})

    # have the agents compete for 30 rounds
    results = ts.compete(jung, skinner, p_matrix=penny, n_rounds=4)

    # examine results
    print(results.head())  # inspect the first 5 rows of the dataframe

    # Creating a simple 1-ToM with default parameters
    tom_1 = ts.TOM(level=1, dilution=None, save_history=True)

    # Extract the parameters
    tom_1.print_parameters()

    tom_2 = ts.TOM(
        level=2,
        volatility=-2,
        b_temp=-2,  # more deterministic
        bias=0,
        dilution=None,
        save_history=True,
    )
    choice = tom_2.compete(p_matrix=penny, agent=0, op_choice=None)
    print("tom_2 choose:", choice)

    tom_2.reset()  # reset before start

    prev_choice_1tom = None
    prev_choice_2tom = None
    for trial in range(1, 4):
        # note that op_choice is choice on previous turn
        # and that agent is the agent you respond to in the payoff matrix
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

    tom_2.print_internal(
        keys=["p_k", "p_op"], level=[0, 1]  # print these two states
    )  # for the agent simulated opponents 0-ToM and 1-ToM

    # Create a list of agents
    agents = ["RB", "QL", "WSLS", "1-TOM", "2-TOM"]
    # And set their starting parameters. An empty dictionary denotes default values
    start_params = [{"bias": 0.7}, {"learning_rate": 0.5}, {}, {}, {}]

    group = ts.create_agents(agents, start_params)  # create a group of agents

    # Specify the environment
    # round_robin e.g. each agent will play against all other agents
    group.set_env(env="round_robin")

    # Finally, we make the group compete 20 simulations of 30 rounds
    results = group.compete(p_matrix=penny, n_rounds=4, n_sim=2, save_history=True)

    res = group.get_results()
    print(res.head(1))  # print the first row

    res.head(1)

    # res.to_json("tutorials/paper.ndjson", orient="records", lines=True)

    import matplotlib.pyplot as plt

    # Set figure size
    plt.rcParams["figure.figsize"] = [10, 10]

    # plot a heatmap of the rewards for all agent in the tournament
    group.plot_heatmap(cmap="RdBu", show=False)

    # plot the choices of the RB agent when competing against the Q-learning agent
    group.plot_choice(
        agent0="RB", agent1="QL", agent=0, plot_individual_sim=False, show=False
    )

    # plot the score of the RB agent when competing against the Q-learning agent
    group.plot_score(agent0="RB", agent1="QL", agent=0, show=False)

    # plot 2-ToM estimate of its opponent sophistication level
    group.plot_p_k(agent0="1-TOM", agent1="2-TOM", agent=1, level=0, show=False)
    group.plot_p_k(agent0="1-TOM", agent1="2-TOM", agent=1, level=1, show=False)

    # plot 2-ToM estimate of its opponent's volatility while believing the opponent to be level 1.
    group.plot_tom_op_estimate(
        agent0="1-TOM",
        agent1="2-TOM",
        agent=1,
        estimate="volatility",
        level=1,
        plot="mean",
        show=False,
    )

    # plot 2-ToM estimate of its opponent's bias while believing the opponent to be level 1.
    group.plot_tom_op_estimate(
        agent0="1-TOM",
        agent1="2-TOM",
        agent=1,
        estimate="bias",
        level=1,
        plot="mean",
        show=False,
    )
