import numpy as np

import tomsup as ts


def test_PayoffMatrix():
    staghunt = ts.PayoffMatrix(name="staghunt")
    assert staghunt.payoff(choice_agent0=1, choice_agent1=1, agent=0) == 5
    assert staghunt.payoff(choice_agent0=1, choice_agent1=0, agent=0) == 0
    assert staghunt.payoff(choice_agent0=0, choice_agent1=1, agent=0) == 3

    chicken = ts.PayoffMatrix(name="chicken")
    assert chicken.payoff(0, 1, 0) == -1

    dead = ts.PayoffMatrix(name="deadlock")
    assert dead.payoff(1, 0, 1) == 0

    sexes = ts.PayoffMatrix(name="sexes")
    assert sexes.payoff(1, 1, 0) == 5

    custom = ts.PayoffMatrix(
        name="custom", predefined=np.array(([(10, 0), (0, 5)], [(5, 0), (0, 10)]))
    )

    prison = ts.PayoffMatrix(name="prisoners_dilemma")
    assert prison.payoff(choice_agent0=0, choice_agent1=1, agent=0) == 5
    assert prison.payoff(choice_agent0=1, choice_agent1=1, agent=0) == 3
    assert prison.payoff(choice_agent0=0, choice_agent1=0, agent=0) == 1
