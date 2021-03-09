

<h1 align="center">tomsup 👍 Theory of Mind Simulation using Python</h1>


[![PyPI version](https://badge.fury.io/py/tomsup.svg)](https://pypi.org/project/tomsup/)
[![Code style: flake8](https://img.shields.io/badge/Code%20Style-flake8-blue)](https://pypi.org/project/flake8/)
[![pip downloads](https://img.shields.io/pypi/dm/tomsup.svg)](https://pypi.org/project/tomsup/)
[![python versions](https://img.shields.io/pypi/pyversions/tomsup?colorB=blue)](https://pypi.org/project/tomsup/)

A Python Package for Agent Based simulations. The package provides a computational eco-system for investigating and comparing computational models of hypothesized Theory of mind (ToM) mechanisms and for using them as experimental stimuli. The package notably includes an easy-to-use implementation of the variational Bayesian k-ToM model developed by [Devaine, et al. (2017)](http://dx.plos.org/10.1371/journal.pcbi.1005833). This model has been shown able to capture individual and group-level differences in social skills, including between clinical populations and across primate species. It has also been deemed among the best computational models of ToM in terms of interaction with others and recursive representation of mental states. We provide a series of tutorials on how to implement the k-ToM model and a score of simpler types of ToM mechanisms in game theory based simulations and experimental stimuli, including how to specify custom ToM models, and show examples of how resulting data can be analyzed.

## 🔧 Setup and installation

tomsup supports Python 3.6 or later. We strongly recommend that you install tomsup from pip. If you haven't installed pip you can install it from [the official pip website](https://pip.pypa.io/en/stable/installing/), otherwise simply run 

```bash
pip3 install tomsup 
```

You can also install it directly from github by simply running:
```bash
pip install git+https://github.com/KennethEnevoldsen/tomsup.git
```

or more explicitly:
```bash
git clone https://github.com/KennethEnevoldsen/tomsup.git
cd tomsup
pip3 install -e .
```


## Getting Started with tomsup
To get started with tomsup we recommend the tutorials in the tutorials [folder](https://github.com/KennethEnevoldsen/tomsup/tree/master/tutorials). We recommend that you start with the introduction.

The tutorials are provided as Jupyter Notebooks. If you do not have Jupyter Notebook installed, instructions for installing and running can be found [here]( http://jupyter.org/install). 


| Tutorial                                                                                                                         | Content                                                                                        | file name                                         |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| [Introduction](https://github.com/KennethEnevoldsen/tomsup/blob/master/tutorials/paper_implementation.ipynb)                     | a general introduction to the features of tomsup which follows the implementation in the paper | paper_implementation.ipynb                        |
| [Creating an agent](https://github.com/KennethEnevoldsen/tomsup/blob/master/tutorials/Creating_an_agent.ipynb)                   | an example of how you would create new agent for the package.                                  | Creating_an_agent.ipynb                           |
| [Specifying internal states](https://github.com/KennethEnevoldsen/tomsup/blob/master/tutorials/specifying_internal_states.ipynb) | a short guide on how to specify internal states on a k-ToM agent                               | specifying_internal_states.ipynb                  |
| [Pscyhopy experiment](https://github.com/KennethEnevoldsen/tomsup/tree/master/tutorials/psychopy_experiment)                     | An example of how one might implement tomsup in an experiment                                  | Not a notebook, but a folder, psychopy_experiment |


## ❓ Issues and Usage Q&A

To ask questions, report issues or request features, please use the [GitHub Issue Tracker](https://github.com/KennethEnevoldsen/tomsup/issues).


## Using this Work
### License
tomsup is released under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

### Citing
If you use this work please cite:
```bibtex
@article{enevoldsen2020tomsup,
  title={tomsup: An implementation of computational Theory of Mind in Python},
  author={Enevoldsen, Kenneth C and Waade, Peter Thestrup},
  year={2020},
  publisher={PsyArXiv}
}
```
