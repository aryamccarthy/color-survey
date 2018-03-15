import numpy as np
from numpy.random import choice, rand
from scipy.stats import multivariate_normal as mvn

np.random.seed(1337)


def mvn_rv(dim=3):
    return mvn.rvs(np.zeros(dim))


class AlignmentGibbsSampler(object):
    """docstring for AlignmentGibbsSampler"""
    def __init__(self, prototypes, manifestations):
        super(AlignmentGibbsSampler, self).__init__()
        N = len(prototypes)
        n = len(manifestations)
        assert n < N, f"Cannot align {n} to {N}."
        self.prototypes = prototypes
        self.manifestations = manifestations
        self.N = N
        self.n = n

    @property
    def U(self):
        return self.prototypes

    @property
    def V(self):
        return self.manifestations

    def init_guess(self):
        return choice(self.N, size=self.n, replace=False)

    def _unaligned(self, state):
        return np.array(list(set(range(self.N)) - set(state)))

    def _log_ratio(self, k, q, r):
        p_q = mvn.pdf(self.V[k], self.U[q])
        p_r = mvn.pdf(self.V[k], self.U[r])
        return p_q - np.logaddexp(p_q, p_r)

    def sample(self, max_iters):
        state = self.init_guess()
        # trace = []

        for i in range(max_iters):

            for k in range(self.n):
                new_value = choice(self._unaligned(state))
                old_value = state[k]
                accept_logprob = self._log_ratio(k, new_value, old_value)
                if np.log(rand()) < accept_logprob:
                    state[k] = new_value
            yield state.copy()
            # trace.append(state.copy())
        # return trace


if __name__ == '__main__':
    n_prototypes = 100
    n_manifestations = 20

    prototypes = [mvn_rv(3) for _ in range(n_prototypes)]
    manifestations = [mvn_rv(3) for _ in range(n_manifestations)]

    for state in AlignmentGibbsSampler(prototypes, manifestations).sample(50):
        print(state)
        # print('[' + ", ".join(str(x) for x in state) + ']')
