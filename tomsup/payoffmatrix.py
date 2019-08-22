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
    >>> p_matrix.payoff(action_agent0 = 1, action_agent1 = 1 , agent = 0)
    1
    >>> p_matrix = PayoffMatrix(name="staghunt")
    >>> p_matrix.payoff(action_agent0 = 1, action_agent1 = 1 , agent = 0)
    5
    >>> p_matrix.payoff(action_agent0 = 1, action_agent1 = 0 , agent = 0)
    0
    >>> p_matrix.payoff(action_agent0 = 0, action_agent1 = 1 , agent = 0)
    3
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
            self.matrix = np.array(([(1, -1),
                                     (-1, 1)],
                                    [(-1, 1),
                                     (1, -1)]))
        if name == "staghunt":
                          #choice a1: 0  1 - Choice a0
            self.matrix = np.array(([(3, 3), # 0   --   Payoff matrix for a0
                                     (0, 5)],# 1
                                    [(3, 0), #     --   Payoff matrix for a1
                                     (3, 5)]))
        if name == "party":
            self.matrix = np.array(([(5, 0), 
                                     (0, 10)],
                                    [(5, 0), 
                                     (0, 10)]))
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


if __name__ == "__main__":
  import doctest
  doctest.testmod(verbose=True)

