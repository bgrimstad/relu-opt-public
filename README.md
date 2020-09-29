# Project description

Example code for optimizing MILP problems with embedded ReLU networks. 
The code is an extraction of the code used in the paper:

```
@article{Grimstad2019,
    author = {Grimstad, Bjarne and Andersson, Henrik},
    journal = {Computers and Chemical Engineering},
    title = {{ReLU Networks as Surrogate Models in Mixed-Integer Linear Programs}},
    pages = {106580},
    volume = {131},
    year = {2019},
    doi = {10.1016/j.compchemeng.2019.106580}
}

```

The code shows how to:
- Load ReLU networks trained with TensorFlow
- Program ReLU networks as MILPs using Gurobi
- Optimize a production optimization problem with ReLU network constraints (see a description of the case in the paper)


### Setup (Linux)

1. Install Anaconda (see https://www.anaconda.com/)

2. Create Conda environment: ``conda env create -f environment.yml``

3. Get the Gurobi license (grbgetkey is located in the Conda environment's bin folder): ``~/anaconda3/envs/relu-opt/bin/grbgetkey <key identifier>``


### How to run

Start the optimization by running the script: ``prodopt/solve_problem.py``
