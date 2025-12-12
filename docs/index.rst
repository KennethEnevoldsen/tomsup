tomsup
================================

.. image:: https://img.shields.io/github/stars/KennethEnevoldsen/tomsup.svg?style=social&label=Star&maxAge=2592000
   :target: https://github.com/KennethEnevoldsen/tomsup

This website contains the documentation for tomsup. tomsup is a Python package for agent-based simulations. The package provides a computational eco-system for investigating and comparing
computational models of hypothesized Theory of mind (ToM) mechanisms and for using them as experimental stimuli. The package notably
includes an easy-to-use implementation of the variational Bayesian k-ToM model developed by `Devaine, et al. (2017) <http://dx.plos.org/10.1371/journal.pcbi.1005833>`__. 
This model has been shown able to capture individual and group-level differences in social skills, including between clinical populations
and across primate species. It has also been deemed among the best computational models of ToM in terms of interaction with others and
recursive representation of mental states. We provide a series of tutorials on how to implement the k-ToM model and a score of simpler
types of ToM mechanisms in game-theory based simulations and experimental stimuli, including how to specify custom ToM models, and show
examples of how resulting data can be analyzed.


📰 News
---------------------------------

* Version 1.4.0

  - Major improvements to ensure long-term maintainance of the project

    - Added support for >3.9 and dropped support for =<3.8

    - Updated installations to use the new `pyproject.toml` standard

    - resolved multiple deprecation warnings

    - Allow for numpy and pandas dependencies >2.0.0

    - Added `.lock` file to ensure reproducibility

* 7 March 2022

  - Paper accepted at `Behavior Research Methods <https://link.springer.com/article/10.3758/s13428-022-01827-2>`__ (2022)

* Version 1.1.5
  
  - New plotting features were added
  
  - Speed and memory improvements as well as support for multicore simulations 🏎
  
  - Added workflows to ensure dependencies are being kept up to date

  - Minor bugfixes 

* Version 1.1.0

  - A `speed comparison <missing>`__ between the matlab implementation was introduced, showing the the tomsup implementation to be notably faster.

  - An extensive testsuite was introduced, for how to run it see the FAQ.

  - A `documentation <https://KennethEnevoldsen.github.io/tomsup/>`__ site was introduced.

  - Continiuous integration to ensure that the package always works as intended.

  - A new logo was introduced 🌟

* Version 1.0.0

  - tomsup released its first version along with a `preprint <https://psyarxiv.com/zcgkv/>`__ on psyarxiv

  - A series of `tutorials <https://KennethEnevoldsen.github.io/tomsup/using-tomsup>`__ was introduced to get you started with tomsup


Contents
---------------------------------
  
The documentation is organized into two parts:

- **Getting Started** contains the installation instructions, guides, and tutorials on how to use tomsup.
- **Package References** contains the documentation of each public class and function. Use this for looking into specific functions.

.. toctree::
   :maxdepth: 3
   :caption: Getting Started

   installation
   using-tomsup

.. toctree::
   :maxdepth: 3
   :caption: Package References

   agent
   payoffmatrix
   plot
   utils
   ktom_functions


.. toctree::
  GitHub Repository <https://github.com/KennethEnevoldsen/tomsup>



Indices and search
==================

* :ref:`genindex`
* :ref:`search`
