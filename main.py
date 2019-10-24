import argparse

import numpy as np
import logging
from mpi4py import MPI

import graph
import dataset
import logistic_regression
import adfs
from utils import plot, get_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'To run this file, run mpirun -n nb_nodes python main.py '
        'with nb_nodes the number of processors on the 2D grid (4, 9, 16...)')

    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument(
        '--nb_points_per_node', type=int, default=300, help='Points per node')
    parser.add_argument(
        '--d', type=int, default=30, help='Dimension of the features')
    parser.add_argument(
        '--c', type=float, default=0.1, help='Regularization per node')
    parser.add_argument(
        '--tau', type=float, default=5, help='Communication delay')
    parser.add_argument(
        '--n_steps',
        type=int,
        default=50000,
        help='Number of algorithm iterations')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    node_id = comm.Get_rank()
    nb_nodes = comm.size

    log = get_logger(node_id)

    # Initialize the graph (grid with nb_nodes nodes)
    grid_graph = graph.Grid(nb_nodes, args.seed, tau=args.tau)

    # Initialize the synthetic dataset
    dataset = dataset.ClassificationDataset(
        seed=args.seed, nb_points=args.nb_points_per_node * nb_nodes, d=args.d)

    # Initialize the model
    model = logistic_regression.LogisticRegression(dataset, nb_nodes * args.c)

    # Initialize the algorithm
    algo = adfs.ADFS(
        comm=comm, seed=args.seed, graph=grid_graph, model=model, log=log)

    # Run the algorithm
    algo.run(args.n_steps)

    # Plot the error
    if node_id == 0:
        min_error = min(algo.error)
        plot(algo.time, algo.error, min_error)
