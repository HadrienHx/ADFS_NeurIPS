import matplotlib.pyplot as plt
import numpy as np
import logging
import seaborn as sns
sns.set(color_codes=True)


def plot(time, error, min_error):
    suboptimality = np.array(error) - min_error
    plt.plot(time, suboptimality)
    plt.yscale("log")
    plt.ylabel("Mean error")
    plt.xlabel("Time")
    plt.show()


def get_logger(node_id):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("node_%d" % node_id)
    if node_id > 0:
        log.setLevel(logging.WARNING)
    return log


def mpi_reduce(comm, data, op=None):
    node_id = comm.Get_rank()
    sendbuf = np.array(data)
    recvbuf = None
    if node_id == 0:
        recvbuf = np.empty((len(data), ), dtype=np.float)

    comm.Reduce(sendbuf, recvbuf, root=0, op=op)

    if node_id == 0:
        return recvbuf
