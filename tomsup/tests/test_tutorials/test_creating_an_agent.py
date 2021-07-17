import tomsup as ts


def test_tutorial():
    sigmund = ts.WSLS()  # create agent

    # inspect sigmund
    print(f"sigmund is an class of type: {type(sigmund)}")  # f is for format
    if isinstance(sigmund, ts.Agent):
        print(f"but sigmund is also of has the parent class ts.Agent")

    class ReversedWSLS(ts.Agent):  # make sure that the parent class is ts.Agent
        """
        ReversedWSLS: Win-switch, lose-stay.

        This agent is a reversed win-stay, lose-switch agent, which ...
        """

        # add a docstring which explains the agent
        pass  # we will later replace this pass with something else

    freud = ReversedWSLS()
    print(f"is freud an Agent? {isinstance(freud, ts.Agent)}")

    class ReversedWSLS(ts.Agent):
        """
        ReversedWSLS: Win-switch, lose-stay.

        This agent is a reversed win-stay, lose-switch agent, which ...
        """

        def __init__(self, first_move, **kwargs):  # initalize the agent
            self.strategy = "ReversedWSLS"  # set the strategy name

            # set internal parameters
            self.first_move = first_move

            super().__init__(
                **kwargs
            )  # pass additional argument the ts.Agent class (could e.g. include 'save_history = True')
            self._start_params = {
                "first_move": first_move,
                **kwargs,
            }  # save any starting parameters used when the agent is reset

    freud = ReversedWSLS(first_move=1)
    print(f"what is freud's first move? {freud.first_move}")
    print(f"what is freud's an starting parameters? {freud.get_start_params()}")
    print(f"what is freud's strategy? {freud.get_strategy()}")

    class ReversedWSLS(ts.Agent):
        """
        ReversedWSLS: Win-switch, lose-stay.

        This agent is a reversed win-stay, lose-switch agent, which ...
        """

        def __init__(self, first_move, **kwargs):  # initalize the agent
            self.strategy = "ReversedWSLS"  # set the strategy name

            # set internal parameters
            self.first_move = first_move

            super().__init__(
                **kwargs
            )  # pass additional argument the ts.Agent class (could e.g. include 'save_history = True')
            self._start_params = {
                "first_move": first_move,
                **kwargs,
            }  # save any starting parameters used when the agent is reset

        def compete(self, p_matrix, op_choice=None, agent=0):
            """
            win-switch, lose-stay strategy, with the first move being set when the class is initilized (__init__())

            p_matrix is a PayoffMatrix
            op_choice is either 1 or 0
            agent is either 0 or 1 and indicated the perpective of the agent in the game (whether it is player 1 og 2)
            """
            if (
                self.choice is None
            ):  # if a choice haven't been made: Choose the redifined first move
                self.choice = self.first_move  # fetch from self
            else:  # if a choice have been made:
                payoff = p_matrix.payoff(
                    self.choice, op_choice, agent
                )  # calculate payoff of last round
                if payoff == 1:  # if the agent won then switch
                    self.choice = (
                        1 - self.choice
                    )  # save the choice in self (for next round)
                    # also save any other internal states which you might
                    # want the agent to keep for next round in self
            self._add_to_history(
                choice=self.choice
            )  # save action and (if any) internal states in history
            # note that _add_to_history() is not intented for
            # later use within the agent
            return self.choice  # return choice which is either 1 or 0

    freud = ReversedWSLS(first_move=1)  # create the agent

    # fetch payoff matrix for the pennygame
    penny = ts.PayoffMatrix(name="penny_competitive")
    print(
        "This is the payoffmatrix for the game (seen from freud's perspective):",
        penny()[0, :, :],
        sep="\n",
    )

    # have freud compete
    choice = freud.compete(penny)
    print(f"what is freud's choice the first round? {choice}")
    choice = freud.compete(penny, op_choice=1)
    print(f"what is freud's choice the second round if his opponent chose 1? {choice}")

    class ReversedWSLS(ts.Agent):
        """
        ReversedWSLS: Win-switch, lose-stay.

        This agent is a reversed win-stay, lose-switch agent, which ...

        Examples:
        >>> waade = ReversedWSLS(first_move = 1)
        >>> waade.compete(op_choice = None, p_matrix = penny)
        1
        """

        def __init__(self, first_move, **kwargs):
            self.strategy = "ReversedWSLS"

            # set internal parameters
            self.first_move = first_move

            super().__init__(
                **kwargs
            )  # pass additional argument the ts.Agent class (could e.g. include 'save_history = True')
            self._start_params = {
                "first_move": first_move,
                **kwargs,
            }  # save any starting parameters used when the agent is reset

        def compete(self, p_matrix, op_choice=None):
            if (
                self.choice is None
            ):  # if a choice haven't been made: Choose the redifined first move
                self.choice = self.first_move  # fetch from self
            else:  # if a choice have been made:
                payoff = p_matrix.payoff(
                    self.choice, op_choice, 0
                )  # calculate payoff of last round
                if payoff == 1:  # if the agent won then switch
                    self.choice = (
                        1 - self.choice
                    )  # save the choice in self (for next round)
                    # also save any other internal states which you might
                    # want the agent to keep for next round in self
            self._add_to_history(
                choice=self.choice
            )  # save action and (if any) internal states in history
            # note that _add_to_history() is not intented for
            # later use within the agent
            return self.choice  # return choice

        # define any additional function you wish the class should have
        def get_first_move(self):
            return self.first_move
