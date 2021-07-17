"""
This scripts contains the PayoffMatrix
"""
from typing import Optional
import numpy as np


class PayoffMatrix:
    """
    A class of 2 by 2 payoff matrices.

    Currently include the following games:
    The staghunt game: 'staghunt',
    The matching pennies game (coop and competive): 'penny_competive', 'penny_cooperative',
    The party dilemma: 'party',
    The Battle of the sexes: 'sexes',
    The chicken game: 'chicken',
    The deadlock: 'deadlock',
    The prisoners dilemma: 'prisoners_dilemma'.

    For custom payoff matrix supply a 2x2x2 numpy array to the predefined argument.

    Examples:
        >>> import tomsup as ts
        >>> staghunt = ts.PayoffMatrix(name="staghunt")
        >>> staghunt.payoff(action_agent0=1, action_agent1=1, agent=0)
        5
        >>> staghunt.payoff(action_agent0=1, action_agent1=0, agent=0)
        0
        >>> staghunt.payoff(action_agent0=0, action_agent1=1, agent=0)
        3
        >>> chicken = ts.PayoffMatrix(name="chicken")
        >>> chicken.payoff(0, 1, 0)
        -1
        >>> dead = ts.PayoffMatrix(name="deadlock")
        >>> dead.payoff(1, 0, 1)
        0
        >>> sexes = ts.PayoffMatrix(name="sexes")
        >>> sexes.payoff(1, 1, 0)
        5
        >>> custom = ts.PayoffMatrix(name="custom", np.array(([(10, 0), (0, 5)],
                                                        [(5, 0), (0, 10)])))
    """

    def __init__(self, name: str, predefined: Optional[np.array] = None):
        """
        Args:
            name (str): The name of the either predefined matrix or your custom matrix.
                Currently include the following games:
                The staghunt game ('staghunt'),
                the matching pennies game ('penny_competive', 'penny_cooperative'),
                the party dilemma ('party'),
                the Battle of the sexes ('sexes'), the chicken game ('chicken'),
                the deadlock ('deadlock'), nad the prisoners dilemma ('prisoners_dilemma').
            predefined (Optional[np.array], optional): A custom 2x2x2 matrix. Defaults to None.
        """
        self.name = name
        if name == "staghunt":
            self.matrix = np.array(
                (
                    [(3, 3), (0, 5)],
                    [(3, 0), (3, 5)],
                )
            )
        elif name == "prisoners_dilemma":
            self.matrix = np.array(([(1, 5), (0, 3)], [(1, 0), (5, 3)]))
        elif name == "penny_competitive":
            self.matrix = np.array(([(-1, 1), (1, -1)], [(1, -1), (-1, 1)]))
        elif name == "penny_cooperative":
            self.matrix = np.array(([(1, -1), (-1, 1)], [(1, -1), (-1, 1)]))
        elif name == "party":
            self.matrix = np.array(([(5, 0), (0, 10)], [(5, 0), (0, 10)]))
        elif name == "sexes":  # battle of the sexes
            self.matrix = np.array(([(10, 0), (0, 5)], [(5, 0), (0, 10)]))
        elif name == "chicken":
            self.matrix = np.array(([(-1000, -1), (1, 0)], [(-1000, 1), (-1, 0)]))
        elif name == "deadlock":
            self.matrix = np.array(([(1, 0), (3, 2)], [(1, 3), (0, 2)]))
        else:
            if predefined is not None:
                matrix = np.array(predefined)
                if matrix.shape == (2, 2, 2):
                    self.matrix = np.array(predefined)
                else:
                    raise TypeError(
                        "Predefined should be a valid matrix where \
                                     matrix.shape == (2, 2, 2), e.g. a 2x2x2 \
                                     matrix"
                    )
            else:
                raise TypeError(
                    "Invalid name and no predefined matrix given. \
                                 Please input a valid name or input a \
                                 predefined matrix of dimension 2x2x2."
                )

    def payoff(self, choice_agent0: int, choice_agent1: int, agent: int = 0) -> float:
        """
        Args:
            choice_agent0 (int): choice of agent 0
            choice_agent1 (int): choice of agent 1
            agent (int, optional): The perspective agent which should get the payoff, either 0 or 1.
                Defaults to 0.

        Returns:
            float: The payoff of the agent
        """

        return self.matrix[agent, choice_agent0, choice_agent1]

    def __str__(self):
        print_len = max([len(str(self().min())), len(str(self().max()))])

        def add_pl(string):
            return (print_len - len(str(string))) * " " + str(string)

        str1 = f"<Class PayoffMatrix, Name = {self.name}> "
        str2 = "The payoff matrix of agent 0"
        str3 = "       |  Choice agent 1"
        str4 = (
            "       | "
            + f"{add_pl(' ')}"
            + " | "
            + f"{add_pl(0)}"
            + " | "
            + f"{add_pl(1)}"
            + " |"
        )
        str5 = "       | " + print_len * 3 * "-" + 2 * "---" + " |"
        str6 = (
            "Choice | " + f"{add_pl(0)}" + " | " + f"{add_pl(self()[0][0,0])}" + " | "
            f"{add_pl(self()[0][0,1])}" + " |"
        )
        str7 = (
            "agent 0| " + f"{add_pl(1)}" + " | " + f"{add_pl(self()[0][1,0])}" + " | "
            f"{add_pl(self()[0][1,1])}" + " |"
        )
        str8 = " "
        str9 = "The payoff matrix of agent 1"
        str10 = "       |  Choice agent 1"
        str11 = (
            "       | "
            + f"{add_pl(' ')}"
            + " | "
            + f"{add_pl(0)}"
            + " | "
            + f"{add_pl(1)}"
            + " |"
        )
        str12 = str5
        str13 = (
            "Choice | " + f"{add_pl(0)}" + " | " + f"{add_pl(self()[1][0,0])}" + " | "
            f"{add_pl(self()[1][0,1])}" + " |"
        )
        str14 = (
            "agent 0| " + f"{add_pl(1)}" + " | " + f"{add_pl(self()[1][1,0])}" + " | "
            f"{add_pl(self()[1][1,1])}" + " |"
        )
        str15 = str8
        return "\n".join(
            [
                str1,
                str2,
                str3,
                str4,
                str5,
                str6,
                str7,
                str8,
                str9,
                str10,
                str11,
                str12,
                str13,
                str14,
                str15,
            ]
        )

    def get_matrix(self) -> np.array:
        """
        Returns:
            np.array: The payoff matrix
        """
        return self.matrix

    def __call__(self):
        return self.matrix
