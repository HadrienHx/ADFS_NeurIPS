## Code to run the ADFS algorithm

This repository contains example code of the ADFS algorithm introduced in the paper "An Accelerated Decentralized Stochastic Proximal Algorithm for Finite Sums", Hadrien Hendrikx, Francis Bach and Laurent Massoulié. 

The code is focused on logistic regression and is voluntarily minimalistic. It is intended to provide a simple example of how ADFS can be implemented to help anyone that would like to use it to recode it. This includes how to use leverage sparse datasets in the linear case.

# Requirements

Install the following packages:

`conda install -c conda-forge openmpi`

`conda install mpi4py seaborn numpy scipy`


# Run the code

To run the code, use the command:

`mpirun -n nb_nodes python main.py`

with nb_nodes the number of processors on the 2D grid (4, 9, 16...)

For instance:

`mpirun -n 4 python main.py`
