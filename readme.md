## Code to run the ADFS algorithm

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
