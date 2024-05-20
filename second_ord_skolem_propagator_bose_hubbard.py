import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import eigh_tridiagonal

# Number theoretical tools: The Skolem polynomials and its inverse
from skolem_util import skolem, tuple_from_skolem

class BHSkolemPropagator(object):
    """
    The second-order split-operator propagator for the 1D Bose-Hubbard model via the Skolem method.
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

        # indx_T implements the shift matrix S^T = S^{-1} in the paper,
        # i.e.,  \Psi[indx_T] implements S^T\Psi
        self.indx_T = np.array([
            skolem(_) for _ in n[np.roll(np.arange(K), minus * -1)].T
        ])
        self.indx_T -= S_min

        ################################################################################################################
        # The following re-ordering is to be compartible with QuSpin
        n = n[::-1]
        # Now the basis in QuSpin and in the Skolem propagator coincide

        ################################################################################################################
        # Get $\exp(-i\Delta t \hat{H}_{nn})$ via block tridiagonalization
        # where $\hat{H}_{nn}$ stands for nearest neighbor hamiltonian
        ################################################################################################################

        # diagonal of the tri-diagonal Hamiltonian H
        d = -(mu + 0.5 * U) * n[-2] + 0.5 * U * n[-2] ** 2

        # off diagonal of the full tri-diagonal Hamiltonian H23
        e = -J * np.sqrt((n[-2] + 1) * n[-1])[1:]

        start_block = 0

        # Find the edges of the blocks. They are found when the off-diagonals have zeros
        block_edges = np.where(e == 0)[0] + 1

        if block_edges[-1] != d.size:
            block_edges = np.append(block_edges, [d.size])

        expH = []

        # Find matrix exponential via diagonalization of each block separately
        for end_block in block_edges:
            block_E, block_v = eigh_tridiagonal(
                d[start_block:end_block],
                e[start_block:end_block - 1],
            )

            expH.append(
                (block_v * np.exp(-1j * 0.5 * dt * block_E)) @ block_v.T
            )


            start_block = end_block

        # Save as a sparse matrix
        self.expH = block_diag(expH)

        ################################################################################################################
        # Prepare for Open boundary condition
        ################################################################################################################
        self.expDiag = np.exp(-1j * dt * d)

    def propagate(self, psi, ntimes = 1, is_pbc = False):
        """
        Evolve the state psi by time self.dt * ntimes
        :param self:
        :param psi: the initial wavefunction
        :param ntimes: number of times steps to propagate
        :param is_pbc: Boundary condition. True for periodic boundary condition, False for open boundary condition.
        :return: np.ndarray - the wavefunction after propagator
        """
        expH = self.expH
        indx = self.indx
        indx_T = self.indx_T

        if is_pbc:
            # Periodic boundary condition

            # 1. rising then 2. falling indices gives better performance          
            # THAN 1.falling then 2. rising indices   WHY ??
            
            for _ in range(ntimes):
                    
                for __ in range(1, self.K + 1):
                    # rising index
                    psi = expH @ psi
                    psi = psi[indx_T]

                for __ in range(1, self.K + 1):
                    # falling index
                    psi = psi[indx]
                    psi = expH @ psi
                    
        else:
            # Open boundary condition
            for _ in range(ntimes):
            
                for __ in range(1, self.K):
                    psi = expH @ psi
                    psi = psi[indx_T]

                psi *= self.expDiag

                for __ in range(1, self.K):
                    psi = psi[indx]
                    psi = expH @ psi

        return psi
