import numpy as np
from mpi4py import MPI
from utils import mpi_reduce


class ADFS(object):
    def __init__(self, comm=None, seed=0, graph=None, model=None, log=None):
        self.graph = graph
        self.global_model = model
        self.rs = np.random.RandomState(seed)
        self.comm = comm
        self.id = comm.Get_rank()
        self.nb_nodes = graph.size
        self.error = [float("inf")]
        self.time = [0.]
        self.error_update_rate = 1000
        self.local_counter = 0
        self.log = log
        self.initialize()

    def initialize(self):
        self.build_virtual_nodes()
        self.get_rate()
        self.initialize_learning_rates()
        self.initialize_parameter_sequences()

    def build_virtual_nodes(self):
        # Get local models for each data point to compute prox, as well as sampling probabilities
        self.local_models, self.virtual_weights, self.is_local_range = self.global_model.get_local_models_and_probas(
            self.id, self.nb_nodes)
        self.nb_samples_per_node = len(self.local_models) - 1
        self.nScomp = np.sum(self.virtual_weights)
        self.normalized_comp_probas = self.virtual_weights / self.nScomp

    def get_rate(self):
        c = self.local_models[0].get_strong_convexity()
        self.kappa = np.sum(
            [model.get_smoothness() for model in self.local_models]) / c
        # Get computation mus
        self.local_mus2 = [
            model.get_smoothness() * self.graph.min_eig / (c * self.kappa)
            for model in self.local_models[1:]
        ]
        self.log.info("Local Condition Number: %f" % self.kappa)
        in_buffer = np.empty((1, ))
        self.comm.Allreduce(np.array([self.kappa]), in_buffer, MPI.MAX)
        max_kappa = in_buffer[0]
        self.comm.Allreduce(np.array([self.kappa]), in_buffer, MPI.MIN)
        min_kappa = in_buffer[0]
        self.sigma_A = 0.5 * self.graph.min_eig / (max_kappa * c)

        p_edge = 1. / len(self.graph.edges)
        # Get the communication rate
        rho_comm_no_p_comm = p_edge * np.sqrt(
            0.5 * c * self.sigma_A / np.max(self.graph.R_eff))
        # Get the computation rate
        rho_comp_no_p_comp = np.sqrt(0.5 * min_kappa / max_kappa) / self.nScomp
        # Adjust p_comp so that the two match
        self.p_comp = max(0.5,
                          1. / (1 + rho_comp_no_p_comp / rho_comm_no_p_comm))

        self.rho = min(rho_comp_no_p_comp * self.p_comp,
                       rho_comm_no_p_comm * (1 - self.p_comp))
        self.log.info("p_comp: %f" % self.p_comp)
        self.log.info("rho: %f" % self.rho)

    def initialize_learning_rates(self):
        p_edge = (1 - self.p_comp) / len(
            self.graph.edges)  # Probability of picking a given edge
        # s[i][0] and s[i][1] are the learning rates used to perform the gradient update for sequences v and y
        # Actually generates it for all edges but only edges incident to self are required
        self.s = [
            np.array([
                self.rho / (p_edge * self.sigma_A),
                np.power(self.rho, 2) * (1 + R_ij / p_edge) /
                (p_edge * self.sigma_A * (1 + self.rho))
            ]).reshape(2, 1) for R_ij in self.graph.R_eff
        ]

        # Same but for computation edges
        self.s_comp = [
            np.array([
                mu * self.rho / (pij * self.p_comp * self.sigma_A),
                (1 + 1. / (pij * self.p_comp)) * mu * np.power(self.rho, 2) /
                (pij * self.p_comp * self.sigma_A * (1 + self.rho))
            ]).reshape(2, 1)
            for pij, mu in zip(self.normalized_comp_probas, self.local_mus2)
        ]

        # Used for convex combinations during local updates
        self.y_coeff = [(1 - np.power(self.rho, 2) / (pij * self.p_comp)) /
                        (1 + self.rho) for pij in self.normalized_comp_probas]
        self.v_coeff = [(1 - self.rho) * self.rho / (pij * self.p_comp *
                                                     (1 + self.rho))
                        for pij in self.normalized_comp_probas]
        self.vt1_coeff = [
            self.rho * (1 + 1. / (pij * self.p_comp)) / (1 + self.rho)
            for pij in self.normalized_comp_probas
        ]

        # Parameters for primal prox updates
        self.nu = [None for _ in range(len(self.s_comp))]

        # Learning rate used for primal prox updates
        for i, (model,
                s) in enumerate(zip(self.local_models[1:], self.s_comp)):
            self.nu[i] = 1. / ((1. / s[0]) - (1. / model.get_smoothness()))
            assert self.nu[i] > 0, "Condition for using primal prox not met, see Appendix B.3"

        # Main matrix used for convex combination updates
        delta = self.rho * (1 - self.rho) / (1 + self.rho)
        self.B = np.array([[1 - self.rho, self.rho], [delta, 1 - delta]])

    def initialize_parameter_sequences(self):
        # vys is the main parameters sequence
        # vys[0] is for the communication node, and vys[1] to vys[m] for virtual nodes 1 to m
        # Since we are in a linear model, only vys[0] is of dimension (2, d), the others are (2, 1)
        self.vys = [np.zeros((2, self.global_model.d))
                    ] + [np.zeros((2, 1)) for model in self.local_models]
        self.last_step = [0 for _ in range(len(self.local_models))]
        self.current_step = 0
        self.current_time = 0.
        self.grad_recvbuf = np.empty((self.global_model.d, ), dtype=np.float)

    def run(self, nb_steps):
        for t in range(nb_steps):
            self.run_step()

            if t % self.error_update_rate == 0:
                self.time.append(self.current_time)
                current_param = self.local_models[0].get_rescaled_parameter(
                    self.vys[0][1])
                self.error.append(
                    self.global_model.compute_error(current_param))
                self.log.info("Loss at iteration %d / %d : %f" %
                              (t, nb_steps, self.error[-1]))

        self.error = mpi_reduce(self.comm, self.error[1:], op=MPI.SUM)
        self.time = mpi_reduce(self.comm, self.time[1:], op=MPI.MAX)

        if self.id == 0:
            self.error /= self.nb_nodes

    # Runs one iteration of ADFS
    def run_step(self):
        self.current_step += 1
        # Computation update with probability p_comp
        if self.rs.rand() < self.p_comp:
            # Randomly choose a sample in the full dataset
            j = self.rs.choice(
                len(self.normalized_comp_probas),
                p=self.normalized_comp_probas)
            # If the sample is in the local dataset of this node
            if self.is_local_range(j):
                # Perform the update
                self.local_optim(j % self.nb_samples_per_node)

        # Communication update otherwise
        else:
            # Sample an edge uniformly at random from the graph
            edge = self.graph.get_random_edge()
            # If the edge is incident to this node
            if edge.is_tip(self.id):
                # Perform a communication update
                self.comm_update(edge)

    # Used to catch up convex combinations that were missed all at once
    def just_in_time_update(self, idx):
        nb_updates = self.current_step - self.last_step[idx] - 1
        self.last_step[idx] = self.current_step
        if nb_updates > 0:
            self.vys[idx] = np.dot(
                np.linalg.matrix_power(self.B, nb_updates), self.vys[idx])

    # Performs a communication update along this edge
    def comm_update(self, edge):
        # Just in time update of the parameter
        self.just_in_time_update(0)
        # Get y_t / sigma_i
        g1 = self.local_models[0].get_rescaled_parameter(self.vys[0][1])
        # COMMUNICATION STEP, exchange gradients with the selected neighbour
        g2 = self.exchange_gradients(g1, edge.get_other(self.id))
        # Update local parameters with the difference of the gradients
        self.vys[0] = np.dot(self.B, self.vys[0]) - np.dot(
            self.s[edge.idx], (g1 - g2).reshape(1, len(g1)))

    def exchange_gradients(self, grad, dest):
        # Send the local gradient
        self.comm.Send(grad, dest=dest, tag=1)

        # Send current_time to distant node to emulate time
        self.comm.Send(
            np.array([self.current_time + self.graph.get_comm_delay()]),
            dest=dest,
            tag=2)

        recvbuftime = np.empty((1, ), dtype=np.float)
        # Receive distant gradient
        self.comm.Recv(self.grad_recvbuf, source=dest, tag=1)

        # Receive time at which the other node sent the gradient + tau
        self.comm.Recv(recvbuftime, source=dest, tag=2)

        # Sending a gradient is assumed to be instantaneous
        # so the update finishes at time max(current_time, time_received_gradient)
        self.current_time = max(self.current_time, recvbuftime[0])

        # Return distant gradient
        return self.grad_recvbuf

    # Local optimization step, when virtual edge (self.id, j) is picked
    def local_optim(self, j):
        # Catch up missed convex combinations.
        self.just_in_time_update(0)
        self.just_in_time_update(j + 1)

        # Compute new values for y (will be updated later)
        new_yi = self.y_coeff[j] * self.vys[0][1] - self.v_coeff[j] * self.vys[0][0]
        new_yj = (self.y_coeff[j] * self.vys[j + 1][1] -
                  self.v_coeff[j] * self.vys[j + 1][0])

        # Update v_{t+1}^{(i,j)} and v_{t+1}^{(i)}
        self.local_exchange(j)
        self.prox_update(j)

        # Update y with information from the proximal step
        self.vys[0][1] = new_yi + self.vt1_coeff[j] * self.vys[0][0]
        self.vys[j + 1][1] = new_yj + self.vt1_coeff[j] * self.vys[j + 1][0]

        # Update time (this step takes time 1)
        self.current_time += self.graph.get_comp_delay()

    def local_exchange(self, j):
        # Compute the parameters update, only the projection on Xi is needed
        g1 = self.local_models[0].get_rescaled_parameter(self.vys[0][1])
        g2 = self.local_models[j
                               + 1].get_rescaled_parameter(self.vys[j + 1][1])
        dg = self.s_comp[j][0] * (
            self.local_models[j + 1].project_on_Xi(g1) - g2)

        # This updates corresponds to z_{t+1}^{(i)} + z_{t+1}^{(i,j)}
        self.vys[0][0] = np.dot(
            self.B[0],
            self.vys[0] + self.vys[j + 1] * self.local_models[j + 1].X)
        # Computing z_{t+1}^{(i,j)} (1D)
        self.vys[j + 1][0] = np.dot(self.B[0], self.vys[j + 1]) + dg

    def prox_update(self, j):
        # Compute v_{t+1}^{(i,j)} (1D)
        v = self.vys[j + 1][0] * self.nu[j] / self.s_comp[j][0]
        self.vys[j + 1][
            0] = v - self.nu[j] * self.local_models[j + 1].get_primal_prox(
                v / self.nu[j], 1. / self.nu[j])

        # Update the center node
        self.vys[0][0] -= self.vys[j + 1][0] * self.local_models[j + 1].X
