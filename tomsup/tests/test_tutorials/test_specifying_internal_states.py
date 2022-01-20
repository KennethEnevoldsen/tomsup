import tomsup as ts


def test_tutorial():
    # Get the competitive penny game payoff matrix
    penny = ts.PayoffMatrix("penny_competitive")

    tom_1 = ts.TOM(level=1)
    init_states = tom_1.get_internal_states()

    init_states["own_states"]["p_k"] = [0.3, 0.7]
    tom_1.set_internal_states(init_states)

    # print the changed states
    tom_1.print_internal()
