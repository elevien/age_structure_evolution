# Code to accompany the manuscript "Evolutionary dynamics in non-Markovian models of microbial populations"

The root folder contains various python notebooks which provide examples of
our code and reproduce figures in the paper. These include

* ``examples.ipynb`` Examples illustrating usage of simulation code

* ``scripts`` This folder contains some scripts to reproduce figures in the paper. I am in the process of moving this code to python notebooks discussed above.
  * ``scripts/generate_fixation_data.py``

* ``sims`` This is where the actualy algorithms to run the simulations are.
  * ``sims/evolutionary_dynamics.py`` This file contains the actual simulation code

  * ``sims/models.py`` Here we define various models, each specified by a divde function which can be used as input to the simulation functions in ``evolutionary_dynamics.py``.
