"""
docstring
"""
import numpy as np



class PayoffMatrix():
    """
    Example:
    >>> staghunt = PayoffMatrix(name="staghunt")
    >>> staghunt.payoff(action_agent0 = 1, action_agent1 = 1 , agent = 0)
    5
    >>> staghunt.payoff(action_agent0 = 1, action_agent1 = 0 , agent = 0)
    0
    >>> staghunt.payoff(action_agent0 = 0, action_agent1 = 1 , agent = 0)
    3
    >>> chicken = PayoffMatrix(name="chicken")
    >>> chicken.payoff(0, 1 , 0)
    -1
    >>> dead = PayoffMatrix(name="deadlock")
    >>> dead.payoff(1, 0, 1)
    0
    >>> sexes = PayoffMatrix(name="sexes")
    >>> sexes.payoff(1, 1, 0)
    5
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
        if name == "staghunt":
                          #choice a1: 0  1 - Choice a0
            self.matrix = np.array(([(3, 3), # 0   --   Payoff matrix for a0
                                     (0, 5)],# 1
                                    [(3, 0), #     --   Payoff matrix for a1
                                     (3, 5)]))
        elif name == "penny_competitive":            
            self.matrix = np.array(([(-1, 1),
                                     (1, -1)],
                                    [(1, -1),
                                     (-1, 1)]))
        elif name == "penny_cooperative":            
            self.matrix = np.array(([(1, -1),
                                     (-1, 1)],
                                    [(1, -1),
                                     (-1, 1)]))
        elif name == "party":
            self.matrix = np.array(([(5, 0), 
                                     (0, 10)],
                                    [(5, 0), 
                                     (0, 10)]))
        elif name == "sexes": #battle of the sexes
            self.matrix = np.array(([(10, 0), 
                                     (0, 5)],
                                    [(5, 0), 
                                     (0, 10)]))
        elif name == "chicken":
            self.matrix = np.array(([(-1000, -1), 
                                     (1, 0)],
                                    [(-1000, 1), 
                                     (-1, 0)]))
        elif name == "deadlock":
            self.matrix = np.array(([(1, 0), 
                                     (3, 2)],
                                    [(1, 3), 
                                     (0, 2)]))
        else:
            if predefined:
                matrix = np.array(predefined)
                if matrix.shape == (2, 2, 2):
                    self.matrix = np.array(predefined)
                else:
                    raise TypeError("Predefined should be a valid matrix where matrix.shape == (2, 2, 2), e.g. a 2x2x2 matrix")
            else:
                raise TypeError("Invalid name and no predefined matrix given. Please input a valid name or input a predefined matrix of dimension 2x2x2.")

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
