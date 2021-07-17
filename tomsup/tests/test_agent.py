import tomsup as ts


def test_Agent():
    rb = ts.Agent("RB", bias=0.7)
    assert isinstance(rb, ts.agent.RB)
    assert rb.bias == 0.7


def test_RB():
    rb = ts.agent.RB(bias=1, save_history=True)
    assert rb.compete() == 1

    for k in ["bias", "save_history"]:
        assert k in rb.get_start_params()

    rb.compete()
    assert sum(rb.get_history(key="choice", format="list")) == 2


def test_WSLS():
    sigmund = ts.agent.WSLS()
    sigmund.choice = 0  # Manually setting choice
    penny = ts.PayoffMatrix(name="penny_competitive")
    assert 0 == sigmund.compete(op_choice=1, p_matrix=penny, agent=0)
    sigmund.choice = 1  # Manually setting choice
    assert 1 == sigmund.compete(op_choice=0, p_matrix=penny, agent=0)


def test_TFT():
    shelling = ts.agent.TFT(copy_prob=1)
    p_dilemma = ts.PayoffMatrix(name="prisoners_dilemma")
    assert 1 == shelling.compete(op_choice=1, p_matrix=p_dilemma)
    assert 0 == shelling.compete(op_choice=0, p_matrix=p_dilemma)


def test_QL():
    ql = ts.agent.QL()
    p_dilemma = ts.PayoffMatrix(name="prisoners_dilemma")
    assert ql.compete(p_matrix=p_dilemma, agent=0, op_choice=None) in [0, 1]


def test_TOM():
    t0 = ts.agent.TOM(level=0, volatility=-2, b_temp=-1)
    t2 = ts.agent.TOM(level=2, volatility=-2, b_temp=-1)
    t2 = ts.agent.TOM(level=2, volatility=-2, b_temp=-1, dilution=0.5, bias=0.3)


def test_AgentGroup():
    group = ts.AgentGroup(agents=["RB"] * 2, start_params=[{"bias": 1}] * 2)
    assert group.agent_names == ["RB_0", "RB_1"]

    RB_0 = group.get_agent("RB_0")  # extract an agent
    assert RB_0.bias == 1  # should naturally be 1, as we specified it

    group.set_env("round_robin")
    result = group.compete(p_matrix="penny_competitive", n_rounds=2, n_sim=3)

    assert result.shape[0] == 2 * 3  # As there is 3 simulations each containing 2 round

    assert result["payoff_agent1"].mean() == 1
    # Given that both agents have always choose 1, it is clear that agent1
    # always win, when playing the competitive pennygame


def test_compete():
    import pandas as pd

    rb = ts.agent.RB(bias=0.7)
    wsls = ts.agent.WSLS()
    result = ts.compete(rb, wsls, p_matrix="penny_competitive", n_rounds=10)
    assert isinstance(result, pd.DataFrame)

    result = ts.compete(
        rb, wsls, p_matrix="penny_competitive", n_rounds=10, n_sim=3, return_val="list"
    )
    assert len(result) == 3 * 10

    result = ts.compete(
        rb,
        wsls,
        p_matrix="penny_competitive",
        n_rounds=100,
        n_sim=3,
        return_val="df",
        verbose=True,
    )

    assert result["payoff_agent1"].mean() > 0
    # We see that the WSLS() on average win more than it lose vs. the biased agent (RB)


def test_plot_internal():
    t2 = ts.agent.TOM(level=2)
    wsls = ts.agent.WSLS()

    result = ts.compete(
        t2, wsls, p_matrix="penny_competitive", n_rounds=10, save_history=True
    )

    t2.plot_internal(
        fun=lambda internal_states: internal_states["own_states"]["p_op"], show=False
    )
    # plotting the agent belief about its opponents theory of mind level (p_k)
    # probability of sophistication level k=0
    t2.plot_internal(
        fun=lambda internal_states: internal_states["own_states"]["p_k"][0], show=False
    )
    # probability of sophistication level k=1
    t2.plot_internal(
        fun=lambda internal_states: internal_states["own_states"]["p_k"][1], show=False
    )
