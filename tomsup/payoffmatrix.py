"""
docstring
"""
import numpy as np



class PayoffMatrix():
    """
    Example:
    >>> p_matrix = PayoffMatrix(name="penny_competitive")
    >>> p_matrix()
    array([[[ 1, -1],
        [-1,  1]],

       [[-1,  1],
        [ 1, -1]]])
    >>> p_matrix.payoff( action_agent0 = 1, action_agent1 = 1 , agent = 0)
    1

    TODO: 
    - add method to class payoff matrix to fetch value
    - make a nicer print
    """
    def __init__(self, name, predefined=None):
        """
        TODO:
        # add the remaining payoff matrices
        """
        self.name = name
        if name == "penny_competitive":
            self.matrix = np.array(([(1, -1), (-1, 1)],
                                    [(-1, 1), (1, -1)]))
        if predefined:
            matrix = np.array(predefined)
            if matrix.shape == (2, 2, 2):
                self.matrix = np.array(predefined)
            else:
                raise Exception("Predefined should be a valid matrix where matrix.shape == (2, 2, 2), e.g. a 2x2x2 matrix")

    def payoff(self, action_agent0, action_agent1, agent = 0):
        """
        assumes action_agent0 and action_agent1 to be integers
        agent is either 'p0' or 'p1' indicating whether the agent is player one or two
        TODO: update docstring
        """

        return self.matrix[agent, action_agent0, action_agent1]


    def __call__(self):
        return self.matrix

