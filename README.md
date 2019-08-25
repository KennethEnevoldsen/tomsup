# tomsup: Theory of Mind Simulation using Python
A Python Package for Agent Based simulations.

<!--  # NOT RENDERED
[![PyPI Version](link missing)
![Python Versions](link missing)
-->

An implementation of game theory of mind in a agent based framework following the implementation of [Devaine, et al. (2017)](http://dx.plos.org/10.1371/journal.pcbi.1005833).

<!--  # NOT RENDERED
### References
```bibtex
@inproceedings{bibtextag,
 author = {Enevoldsen and Waade},
 title = {Unknown},
 month = {Unkown},
 pages = {Unknown},
 publisher = {Unknown},
 title = {Unknown},
 url = {Unknown},
 year = {2019}
}
```
-->
## Issues and Usage Q&A

To ask questions, report issues or request features, please use the [GitHub Issue Tracker](https://github.com/KennethEnevoldsen/tomsup/issues).

## Setup

tomsup supports Python 3.6 or later. We strongly recommend that you install tomsup from Github as it is the most recent version. This is done by running the following:
```bash
git clone https://github.com/KennethEnevoldsen/tomsup.git
cd tomsup
pip3 install -e .
```

Assuming you have pip install 
However you can also install from pip. If you haven't installed pip you can install it from [here](https://pip.pypa.io/en/stable/installing/), otherwise simply run:
```bash
pip3 install tomsup
```


## Getting Started with tomsup

```python
import tomsup

## Get a list of valid agents
tomsup.valid_agents()
```

Which should get you the output
```
(not yet inserted)
```

```python
## tomsup - create a single agent
    # There is two ways to setup an agent (these are equivalent)
sirRB = tomsup.Agent(strategy = "RB", save_history = True, bias = 0.7) # calling the Agent class specifying strategy
sirRB = tomsup.RB(bias = 0.7, save_history = True) #calling the agent subclass 
isinstance(sirRB, tomsup.Agent)  # sirRB is an Agent 
type(sirRB)  # of supclass RB
```
Which should get you the output
```
(not yet inserted)
```

```python
choice = sirRB.compete()

print(f"SirRB chose {choice} and his probability for choosing 1 was {sirRB.get_bias()}.")
```

Which should e.g. return
```
SirRB chose 1 and his probability for choosing 1 was 0.7.
```

<!--  # NOT RENDERED
### A todolist:

Final things:
- check function text
- check everything works as intended

Need to have
- rework of WSLS
- add TFT

Nice to have:
- Smart initialize ToM
- reinforcement learner agent
-->

## LICENSE
tomsup is released under the Apache [License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

