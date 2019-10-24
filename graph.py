import numpy as np


class Edge(object):
    def __init__(self, idx, i, j):
        self.idx = idx
        self.i = i
        self.j = j

    def is_tip(self, u):
        return (self.i == u) or (self.j == u)

    def get_other(self, u):
        if self.i == u:
            return self.j
        if self.j == u:
            return self.i
        return None


class Grid(object):
    def __init__(self, size, seed=0, tau=1.):
        self.seed = seed
        self.size = size
        self.tau = tau

        self.edges_set = set()
        self.edges = []
        self.rs = np.random.RandomState(seed)

        # Check that the size is compatible with a 2D grid
        sqrt_l = int(np.floor(np.sqrt(size)))
        l2 = int(np.power(sqrt_l, 2))
        assert (l2 == size)

        self.build_edges(sqrt_l)

        # Constructs the laplacian to get the eigengap
        self.build_laplacian()

    def add_edge(self, i, j):
        if j < i:
            i, j = j, i

        if (i, j) not in self.edges_set:
            self.edges_set.add((i, j))
            self.edges.append(Edge(len(self.edges), i, j))

    # Build the edges for a 2D grid
    def build_edges(self, length):
        neighbours_x = ([
            -1, 1
        ], lambda current, delta: -1 < (current % length) + delta < length)
        neighbours_y = ([
            -length, length
        ], lambda current, delta: -1 < current + delta < self.size)
        for i in range(self.size):
            for neighbors, condition in (neighbours_x, neighbours_y):
                for neigh in neighbors:
                    if condition(i, neigh):
                        self.add_edge(i, i + neigh)

    # Assumes mu = 1 for communication edges
    def build_laplacian(self):
        A = np.zeros((self.size, len(self.edges)), dtype=np.double)

        for edge in self.edges:
            A[edge.i, edge.idx] = 1
            A[edge.j, edge.idx] = -1

        AcrossA = np.dot(np.linalg.pinv(A), A)
        self.R_eff = [np.real(AcrossA[i, i]) for i in range(len(AcrossA))]

        w, _ = np.linalg.eigh(np.dot(A, A.T))
        w = np.sort([np.real(x) for x in w])

        # The Laplacian has 1 zero eigenvalue
        self.min_eig = w[1]

    # Communication edges are chosen uniformly at random
    def get_random_edge(self):
        return self.rs.choice(self.edges)

    # Communication delay is tau
    def get_comm_delay(self):
        return self.tau

    # Computation delay is 1
    def get_comp_delay(self):
        return 1.
