import numpy as np
import torch as th

from numpy.random import choice, rand
# from scipy.stats import multivariate_normal as mvn

from torch.distributions import MultivariateNormal
from torch import nn
from tqdm import tqdm, trange

from inverse_nn import invert_our_diffeomorphism

np.random.seed(1337)


DIM = 3


def print(*xs):
    tqdm.write(" ".join([str(x) for x in xs]))


def mvn_rv(*, n_samples=1, dim=DIM):
    mu = th.zeros(dim)
    cov = th.eye(dim)
    mvn = MultivariateNormal(loc=mu, covariance_matrix=cov)
    return mvn.sample(th.Size([n_samples])).detach().numpy()


class AlignmentGibbsSampler(object):
    """docstring for AlignmentGibbsSampler"""
    def __init__(self, prototypes, manifestations, inverse_map):
        super(AlignmentGibbsSampler, self).__init__()
        N = len(prototypes)
        n = len(manifestations)
        assert n < N, f"Cannot align {n} to {N}."
        self.chromemes = prototypes
        self.colors = manifestations
        self.N = N
        self.n = n
        self.inverter = inverse_map

    @property
    def U(self):
        return self.chromemes

    @property
    def V(self):
        try:
            return self.chromes
        except AttributeError:
            self.chromes = [self.inverter(th.Tensor(c)).detach().numpy() for c in self.colors]
            return self.chromes

    def init_guess(self):
        try:
            return self.state
        except AttributeError:
            return choice(self.N, size=self.n, replace=False)

    def _unaligned(self, state):
        return np.array(list(set(range(self.N)) - set(state)))

    def _log_ratio(self, k, q, r):
        U_q = th.Tensor(self.U[q])
        U_r = th.Tensor(self.U[r])
        V_k = th.Tensor(self.V[k])
        mvn_q = MultivariateNormal(loc=U_q, covariance_matrix=th.eye(DIM))
        mvn_r = MultivariateNormal(loc=U_r, covariance_matrix=th.eye(DIM))
        p_q_log = mvn_q.log_prob(V_k)
        p_r_log = mvn_r.log_prob(V_k)
        p_q_log, p_r_log = p_q_log.detach().numpy(), p_r_log.detach().numpy()
        return p_q_log - np.logaddexp(p_q_log, p_r_log)

    def sample(self, draws, *, take_every_nth=1, progressbar=True, burn_in=0):
        self.chromes = [self.inverter(th.Tensor(c)).detach().numpy() for c in self.colors]

        state = self.init_guess()

        iter_method = trange if progressbar else range

        for i in iter_method(1, 1 + draws * take_every_nth + burn_in):

            for k in range(self.n):
                new_value = choice(self._unaligned(state))
                old_value = state[k]
                accept_logprob = self._log_ratio(k, new_value, old_value)
                if np.log(rand()) < accept_logprob:
                    state[k] = new_value
            self.state = state
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

    from complete import prepare_m_step_data

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

