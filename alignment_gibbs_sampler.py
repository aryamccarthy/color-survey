import numpy as np
import torch as th

from numpy.random import choice, rand
from scipy.stats import multivariate_normal as mvn

from torch.distributions import MultivariateNormal
from tqdm import tqdm, trange

np.random.seed(1337)


def print(x):
    tqdm.write(str(x))


def mvn_rv(*, n_samples=1, dim=3):
    mu = th.zeros(3)
    cov = th.eye(3)
    mvn = MultivariateNormal(loc=mu, covariance_matrix=cov)
    return mvn.sample(th.Size([n_samples]))


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

    def sample(self, draws, *, take_every_nth=1, progressbar=True, burn_in=1000):
        state = self.init_guess()
        # trace = []

        iter_method = trange if progressbar else range

        for i in iter_method(1, 1 + draws * take_every_nth + burn_in):

            for k in range(self.n):
                new_value = choice(self._unaligned(state))
                old_value = state[k]
                accept_logprob = self._log_ratio(k, new_value, old_value)
                if np.log(rand()) < accept_logprob:
                    state[k] = new_value
            if i % take_every_nth == 0 and i > burn_in:
                yield state.copy()
            # trace.append(state.copy())
        # return trace


if __name__ == '__main__':
    n_prototypes = 100
    n_manifestations = 20

    prototypes = mvn_rv(dim=3, n_samples=n_prototypes)
    manifestations = mvn_rv(dim=3, n_samples=n_manifestations)

    for state in AlignmentGibbsSampler(prototypes, manifestations).sample(50, take_every_nth=20):
        print(state)
        # print('[' + ", ".join(str(x) for x in state) + ']')
