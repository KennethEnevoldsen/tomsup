# tomsup: Theory of Mind Simulation using Python
A Python Package for Agent Based simulations.

<!--  # NOT RENDERED
[![PyPI Version](link missing)
![Python Versions](link missing)
-->

An implementation of game theory of mind in a agent based framework following the implementation of [Devaine, et al. (2017)](http://dx.plos.org/10.1371/journal.pcbi.1005833). This package also include a theory of mind. 

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

tomsup supports Python 3.6 or later. We strongly recommend that you install tomsup from pip. If you haven't installed pip you can install it from [here](https://pip.pypa.io/en/stable/installing/), otherwise simply run 

(**the version on pip isn't currently fully updated**)
```bash
pip3 install tomsup 
```

However you can also install it directly from github by simply running:
```bash
git clone https://github.com/KennethEnevoldsen/tomsup.git
cd tomsup
pip3 install -e .
```


## Getting Started with tomsup
To get started with tomsup we recommend to the tutorials in the tutorials folder. We recommend that you start with the introduction.


The tutorials is in a Jupyter Notebooks format if you don't have Jupyter Notebook installed, instructions for installing and running can be found here: http://jupyter.org/install. 

Currently we have the following tutorials
- introduction, a general introduction to the features of tomsup
- Creating_an_agent, an example of how you would create new agent for the package. 
- introduction_to_tom, an introduction to the variational bayes theory of mind model used in tomsup. (**Not currently finished**)

There is also an example psychopy experiment, in which the player can play against the theory of mind agent in the matching pennies task. (**currently not finished**)


## LICENSE
tomsup is released under the Apache [License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

