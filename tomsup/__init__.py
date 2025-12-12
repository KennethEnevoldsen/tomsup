from numpy import exp, log

import tomsup.plot as plot
from tomsup.about import __download_url__, __title__, __version__
from tomsup.agent import *  # noqa
from tomsup.ktom_functions import inv_logit, logit
from tomsup.payoffmatrix import *  # noqa
from tomsup.utils import create_agents, valid_agents

__all__ = [
    "valid_agents",
    "create_agents",
    "plot",
    "logit",
    "inv_logit",
    "__version__",
    "__title__",
    "__download_url__",
    "log",
    "exp",
]
