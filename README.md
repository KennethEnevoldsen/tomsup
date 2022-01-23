<a href="https://github.com/KennethEnevoldsen/tomsup"><img src="https://github.com/KennethEnevoldsen/tomsup/raw/master/img/icon_black2.png" width="190" align="right" /></a>


# tomsup: Theory of Mind Simulation using Python 

[![PyPI version](https://badge.fury.io/py/tomsup.svg)](https://pypi.org/project/tomsup/)
[![pip downloads](https://img.shields.io/pypi/dm/tomsup.svg)](https://pypi.org/project/tomsup/)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![python version](https://img.shields.io/badge/Python-%3E=3.6-blue)](https://github.com/KennethEnevoldsen/tomsup)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/KennethEnevoldsen/tomsup/blob/master/LICENSE)
[![github actions pytest](https://github.com/KennethEnevoldsen/tomsup/actions/workflows/pytest-cov-comment.yml/badge.svg)](https://github.com/KennethEnevoldsen/tomsup/actions)
[![github actions docs](https://github.com/KennethEnevoldsen/tomsup/actions/workflows/documentation.yml/badge.svg)](https://KennethEnevoldsen.github.io/tomsup/)
![github coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KennethEnevoldsen/ba4cb2310c5b370dc2c49d0be0a7e3ec/raw/badge-tomsup-pytest-coverage.json)
[![CodeFactor](https://www.codefactor.io/repository/github/KennethEnevoldsen/tomsup/badge)](https://www.codefactor.io/repository/github/KennethEnevoldsen/tomsup)


A Python Package for Agent-Based simulations. The package provides a computational eco-system for investigating and comparing computational models of hypothesized Theory of mind (ToM) mechanisms and for using them as experimental stimuli. The package notably includes an easy-to-use implementation of the variational Bayesian k-ToM model developed by [Devaine, et al. (2017)](http://dx.plos.org/10.1371/journal.pcbi.1005833). This model has been shown able to capture individual and group-level differences in social skills, including between clinical populations and across primate species. It has also been deemed among the best computational models of ToM in terms of interaction with others and recursive representation of mental states. We provide a series of tutorials on how to implement the k-ToM model and a score of simpler types of ToM mechanisms in game-theory based simulations and experimental stimuli, including how to specify custom ToM models, and show examples of how resulting data can be analyzed.


# 📰 News

- V. 1.1.3
  - New plotting features were added
  - Speed and memory improvements as well as support for multicore simulations 🏎
  - Added workflows to ensure dependencies are being kept up to date
- v. 1.1.0
  - A [speed comparison](missing) between the matlab implementation was introduced, showing the the tomsup implementation to be notably faster.
  - An extensive testsuite was introduced, for how to run it see the FAQ.
  - Code coverage was upped to 86% and code quality was raised to A.
  - A [documentation](https://KennethEnevoldsen.github.io/tomsup/) site was introduced.
  - Added continiuous integration to ensure that the package always works as intended, with support for mac, windows and linux tests.
  - A new logo was introduced 🌟
- v. 1.0.0
  - tomsup released its first version along with a [preprint](https://psyarxiv.com/zcgkv/) on psyarxiv
  - A series of [tutorials](https://KennethEnevoldsen.github.io/tomsup/using-tomsup) was introduced to get you started with tomsup

# 🔧 Setup and installation

tomsup supports Python 3.6 or later. We strongly recommend that you install tomsup from pip. If you haven't installed pip you can install it from [the official pip website](https://pip.pypa.io/en/stable/installing/), otherwise, run:

```bash
pip install tomsup 
```

<details>
  <summary>Detailed instructions</summary>

  You can also install it directly from GitHub by simply running:
  ```bash
  pip install git+https://github.com/KennethEnevoldsen/tomsup.git
  ```

  or more explicitly:
  ```bash
  git clone https://github.com/KennethEnevoldsen/tomsup.git
  cd tomsup
  pip3 install -e .
  ```


</details>


## Getting Started with tomsup
To get started with tomsup we recommend the tutorials in the tutorials [folder](https://github.com/KennethEnevoldsen/tomsup/tree/master/tutorials). We recommend that you start with the introduction.

The tutorials are provided as Jupyter Notebooks. If you do not have Jupyter Notebook installed, instructions for installing and running can be found [here]( http://jupyter.org/install). 


| Tutorial                                                                                                                         | Content                                                                                        | file name                                         | Open with                                                                                                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Documentation](https://kennethenevoldsen.github.io/tomsup/)                                                                     | The documentations of tomsup                                                                   |                                                   |                                                                                                                                                                                                        |
| [Introduction](https://github.com/KennethEnevoldsen/tomsup/blob/master/tutorials/paper_implementation.ipynb)                     | a general introduction to the features of tomsup which follows the implementation in the paper | paper_implementation.ipynb                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KennethEnevoldsen/tomsup/blob/master/tutorials/paper_implementation.ipynb)       |
| [Creating an agent](https://github.com/KennethEnevoldsen/tomsup/blob/master/tutorials/Creating_an_agent.ipynb)                   | an example of how you would create new agent for the package.                                  | Creating_an_agent.ipynb                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KennethEnevoldsen/tomsup/blob/master/tutorials/Creating_an_agent.ipynb)          |
| [Specifying internal states](https://github.com/KennethEnevoldsen/tomsup/blob/master/tutorials/specifying_internal_states.ipynb) | a short guide on how to specify internal states on a k-ToM agent                               | specifying_internal_states.ipynb                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KennethEnevoldsen/tomsup/blob/master/tutorials/specifying_internal_states.ipynb) |
| [Psychopy experiment](https://github.com/KennethEnevoldsen/tomsup/tree/master/tutorials/psychopy_experiment)                     | An example of how one might implement tomsup in an experiment                                  | Not a notebook, but a folder, psychopy_experiment | [![Open in Github](https://img.shields.io/badge/%20-Open%20in%20GitHub-black?style=plastic&logo=github)](https://github.com/KennethEnevoldsen/tomsup/tree/master/tutorials/psychopy_experiment)        |


# 🤔 Issues and Usage Q&A

To ask report issues or request features, please use the [GitHub Issue Tracker](https://github.com/KennethEnevoldsen/tomsup/issues). Otherwise, please use the [discussion Forums](https://github.com/KennethEnevoldsen/tomsup/discussions).

## FAQ

<details>
  <summary>How do I test the code and run the test suite?</summary>


tomsup comes with an extensive test suite. In order to run the tests, you'll usually want to clone the repository and build tomsup from the source. This will also install the required development dependencies and test utilities defined in the requirements.txt.


```
pip install -r requirements.txt
pip install pytest

python -m pytest
```

which will run all the test in the `tomsup/tests` folder.

Specific tests can be run using:

```
python -m pytest tomsup/tests/<DesiredTest>.py
```

**Code Coverage**
If you want to check code coverage you can run the following:
```
pip install pytest-cov

python -m pytest--cov=.
```


</details>




<details>
  <summary>Does tomsup run on X?</summary>

  tomssup is intended to run on all major OS, this includes Windows (latest version), MacOS (Catalina) and the latest version of Linux (Ubuntu). Please note these are only the systems tomsup is being actively tested on, if you run on a similar system (e.g. an earlier version of Linux) the package will likely run there as well.

  
</details>


<details>
  <summary>How is the documentation generated?</summary>

  Tomsup uses [sphinx](https://www.sphinx-doc.org/en/master/index.html) to generate documentation. It uses the [Furo](https://github.com/pradyunsg/furo) theme with a custom styling.

  To make the documentation you can run:
  
  ```
  # install sphinx, themes and extensions
  pip install sphinx furo sphinx-copybutton sphinxext-opengraph

  # generate html from documentations

  make -C docs html
  ```
  
</details>



# Using this Work
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
