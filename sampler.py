import numpy as np
import torch as th

from numpy.random import choice, rand
# from scipy.stats import multivariate_normal as mvn

from torch.distributions import MultivariateNormal
from torch import nn
from tqdm import tqdm, trange

from complete import prepare_m_step_data
from inverse_nn import invert_our_diffeomorphism

np.random.seed(1337)


DIM = 3


def print(*xs):
    tqdm.write(" ".join([str(x) for x in xs]))


def mvn_rv(*, n_samples=1, dim=DIM):
    mu = th.zeros(dim)
    cov = th.eye(dim)
    mvn = MultivariateNormal(loc=mu, covariance_matrix=cov)
    return mvn.sample(th.Size([n_samples]))


class AlignmentGibbsSampler(object):
    """docstring for AlignmentGibbsSampler"""
    def __init__(self, prototypes, manifestations, inverse_map):
        super(AlignmentGibbsSampler, self).__init__()
        N = len(prototypes)
        n = len(manifestations)
        assert n < N, f"Cannot align {n} to {N}."
        self.chromemes = prototypes
        self.colors = manifestations
        self.chromes = [inverse_map(th.Tensor(c)).detach().numpy() for c in self.colors]
        self.N = N
        self.n = n

    @property
    def U(self):
        return self.chromemes

    @property
    def V(self):
        return self.chromes

    def init_guess(self):
        return choice(self.N, size=self.n, replace=False)

    def _unaligned(self, state):
        return np.array(list(set(range(self.N)) - set(state)))

    def _log_ratio(self, k, q, r):
        mvn_q = MultivariateNormal(loc=self.U[q], covariance_matrix=th.eye(DIM))
        mvn_r = MultivariateNormal(loc=self.U[r], covariance_matrix=th.eye(DIM))
        p_q_log = mvn_q.log_prob(self.V[k])
        p_r_log = mvn_r.log_prob(self.V[k])
        p_q_log, p_r_log = p_q_log.detach().numpy(), p_r_log.detach().numpy()
        return p_q_log - np.logaddexp(p_q_log, p_r_log)

    def sample(self, draws, *, take_every_nth=1, progressbar=True, burn_in=0):
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

    diffeomorphism = nn.Sequential(
        nn.Linear(DIM, DIM)
        )
    inverted = invert_our_diffeomorphism(diffeomorphism)

    one_speaker_only, _ = prepare_m_step_data(n_prototypes)
    prototypes = mvn_rv(dim=DIM, n_samples=n_prototypes)

    samplers = [AlignmentGibbsSampler(prototypes, inventory, inverted) for inventory in one_speaker_only]
    for sampler in samplers:
        # Burn in
        sampler.sample(0, burn_in=1000)

    language_samples = []
    for sampler in samplers:
        # print(inventory)
        language_samples.append([])
        for state in sampler.sample(5, take_every_nth=20):
            language_samples[-1].append(state)
    for language in language_samples:
        print(language)

