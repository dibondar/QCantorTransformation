import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import eigh_tridiagonal

# Number theoretical tools: The Skolem polynomials and its inverse
from skolem_util import skolem, tuple_from_skolem

class BHSkolemPropagator(object):
    """
    The first-order split-operator propagator for the 1D Bose-Hubbard model via the Skolem method.
    """
    def __init__(self, *, U, J, N, K, mu = 0, dt = 0.01):
        """
        :param U: On-sight interaction
        :param J: Hopping energy
        :param mu: Chemical potential
        :param N: Number of bosons
        :param K: number of sites
        :param dt: time-step
        """
        ################################################################################################################
        # save the parameters
        self.U = U
        self.J = J
        self.dt = dt
        self.K = K
        ################################################################################################################

        # Using the Skolem for building the propagator  the Bose Hubbard model

        # the cut off of Skolem index
        S_min = skolem((K - 1) * [0] + [N])
        S_max = skolem([N] + (K - 1) * [0])

        ################################################################################################################
        # Generates the K-tuple of nonnegative integers that sum up to N
        #
        # This can be accomplished as
        #
        #    skolem_indx = np.arange(S_min, S_max + 1)
        #    n = np.array(
        #        [tuple_from_skolem(S, K) for S in skolem_indx]
        #    ).T
        #
        # The following code is equivalent, but it runs much faster.
        #
        ################################################################################################################

        def generate_tuples(N, K):
            def generate_tuples_recursive(N, K, prefix=[]):
                if K == 1:
                    if N >= sum(prefix):
                        yield prefix + [N - sum(prefix)]
                else:
                    for i in range(N - sum(prefix), -1, -1):
                        yield from generate_tuples_recursive(N, K - 1, prefix + [i])

            return generate_tuples_recursive(N, K)

        n = list(generate_tuples(N, K))
        n = np.array(n).T[::-1]

        ################################################################################################################
        # Reindexing to implement the permutation matrix
        # to bring the neighboring site to tridiagonal form.
        ################################################################################################################

        # This is included to be compatible with QuSpin
        minus = -1

        # indx implements the shift matrix S in the paper,
        # i.e.,  \Psi[indx] implements S\Psi
        self.indx = np.array([
            skolem(_) for _ in n[np.roll(np.arange(K), minus * 1)].T
        ])
        self.indx -= S_min

        ################################################################################################################
        # The following re-ordering is to be compartible with QuSpin
        n = n[::-1]
        # Now the basis in QuSpin and in the Skolemn propagator coincide

        ################################################################################################################
        # Get $\exp(-i\Delta t \hat{H}_{nn})$ via block tridiagonalization
        # where $\hat{H}_{nn}$ stands for nearest neighbor hamiltonian
        ################################################################################################################

        # diagonal of the tri-diagonal Hamiltonian H_nn
        d = -(mu + 0.5 * U) * n[-2] + 0.5 * U * n[-2] ** 2

        # off diagonal of the full tri-diagonal Hamiltonian H23
        e = -J * np.sqrt((n[-2] + 1) * n[-1])[1:]

        start_block = 0

        # Find the edges of the blocks. They are found when the off-diagonals have zeros
        block_edges = np.where(e == 0)[0] + 1

        if block_edges[-1] != d.size:
            block_edges = np.append(block_edges, [d.size])

        expH_nn = []

        # Find matrix exponential via diagonalization of each block separately
        for end_block in block_edges:
            block_E_nn, block_v = eigh_tridiagonal(
                d[start_block:end_block],
                e[start_block:end_block - 1],
            )

            expH_nn.append(
                (block_v * np.exp(-1j * dt * block_E_nn)) @ block_v.T
            )

            start_block = end_block

        # Save as a sparse matrix
        self.expH_nn = block_diag(expH_nn)

        ################################################################################################################
        # Prepare for Open boundary condition
        ################################################################################################################

        # extra phase factor (due to the on-site interaction)
        # that needs to be taken into account in the case of the open boundary condition
        d_edge = d - (mu + 0.5 * U) * n[-1] + 0.5 * U * n[-1] ** 2

        expH_edge = []

        start_block = 0

        # Find matrix exponential via diagonalization of each block separately
        for end_block in block_edges:
            block_E_nn, block_v = eigh_tridiagonal(
                d_edge[start_block:end_block],
                e[start_block:end_block - 1],
            )

            expH_edge.append(
                (block_v * np.exp(-1j * dt * block_E_nn)) @ block_v.T
            )

            start_block = end_block

        # Save as a sparse matrix
        self.expH_edge = block_diag(expH_edge)

    def propagate(self, psi, ntimes = 1, is_pbc = False):
        """
        Evolve the state psi by time self.dt * ntimes
        :param self:
        :param psi: the initial wavefunction
        :param ntimes: number of times steps to propagate
        :param is_pbc: Boundary condition. True for periodic boundary condition, False for open boundary condition.
        :return: np.ndarray - the wavefunction after propagator
        """
        expH_nn = self.expH_nn
        indx = self.indx

        if is_pbc:
            # Periodic boundary condition
            for _ in range(ntimes):
                for __ in range(self.K):
                    psi = expH_nn @ psi[indx]

        else:
            # Open boundary condition
            for _ in range(ntimes):
                psi = self.expH_edge @ psi[indx][indx]

                for __ in range(2, self.K):
                    psi = expH_nn @ psi[indx]

        return psi