# pylint: disable=missing-docstring,invalid-name
from itertools import product
import logging
from math import pi as π

from pathlib import Path

from numpy import allclose

import torch as th
import torch.optim as optim

from torch.nn import Module


DIM = 3


log = logging.getLogger(Path(__file__).stem)
log.setLevel(logging.DEBUG)


class LEnsembleFactory(object):
    """docstring for LEnsembleFactory"""
    def __init__(self, focalizer):
        super(LEnsembleFactory, self).__init__()
        self.focalizer = focalizer

    @staticmethod
    def kernel(μ, μʹ, ρ):
        σ = 1

        term1 = (2 * ρ) ** (DIM / 2)
        term2 = (2 * π * σ**2) ** ((1 - 2*ρ) * DIM / 2)
        term3 = th.exp(-(ρ) * (th.norm(μ - μʹ) ** 2) / (4 * σ**2))
        return term1 * term2 * term3

    def focalization(self, μ):
        return th.exp(self.focalizer(μ))

    def make(self, μs, use_dispersion=True, ρ=None):
        # log.info("Using DPP with dispersion and focalization")
        if not ρ:
            ρ = th.tensor(0.01)
        N = len(μs)
        focalization = th.zeros(N)
        for i, μ in enumerate(μs):
            focalization[i] = self.focalization(μ)
        if use_dispersion:
            dispersion = th.zeros(N, N)
            for (i, μ), (j, μʹ) in product(enumerate(μs), repeat=2):
                dispersion[i, j] = self.kernel(μ, μʹ, ρ)
            L = dispersion + th.diag(focalization)
        else:
            L = th.diag(focalization)
        assert allclose(L.detach().numpy(), L.t().detach().numpy()), ("Did not produce a symmetric L!\n", L)
        return L


class DeterminantalPointProcess(Module):
    """docstring for DeterminantalPointProcess"""
    def __init__(self, L):
        super(DeterminantalPointProcess, self).__init__()
        assert allclose(L.detach().numpy(), L.t().detach().numpy()), ("L must be symmetric.\n", L)
        self._log_L = th.log(L)
        self.N = L.shape[0]

    @property
    def L(self):
        return th.exp(self._log_L)

    @property
    def log_normalizer(self) -> th.FloatTensor:
        return th.logdet(self.L + th.eye(self.N))

    def log_prob(self, x: th.LongTensor) -> th.FloatTensor:
        if isinstance(x, list):
            x = th.LongTensor(x)
        if len(x) == 0:
            logdet = th.log(th.tensor(1.0))  # pylint: disable=not-callable
        else:
            coordinates = th.LongTensor(x)
            submatrix = self.L[coordinates][:, coordinates]
            logdet = th.logdet(submatrix)
        return logdet - self.log_normalizer

    def forward(self, x) -> th.FloatTensor:  # pylint: disable=arguments-differ
        return self.log_prob(x)


def test_backprop():
    data = [[1],
            [0, 2],
            [1, 2],
            [1],
            [],
            [1],
            [0, 2],
            [2],
            [1],
            [0],
            [2],
            [0, 2],
            [1, 2],
            [0, 1],
            [1],
            [2],
            [0],
            [1],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
            [0, 2],
           ]
    # data = [th.LongTensor(t) for t in data]

    M = th.Tensor([[2,    0.25,    0.5],
                   [0.25,    2,   0.25],
                   [0.5,  0.25,      2]])
    other = th.ones((3, 1), requires_grad=True)
    # M = th.Tensor([[2,     -1,      0],
    #                [-1,     2,     -1],
    #                [0,     -1,    2]])
    optimizer = optim.SGD([other], lr=0.1, momentum=0.9)
    for epoch in range(20):
        optimizer.zero_grad()
        dpp = DeterminantalPointProcess(M + th.exp(th.diag(other)))
        loss = th.tensor(0.0)
        for datum in data:
            log_prob = dpp(datum)
            print(f"\t{float(log_prob)}")
            loss -= log_prob
        print(dpp.L)
        print(float(loss) / len(data))
        print("X" * int(loss.data))
        loss.backward()
        optimizer.step()
    print(dpp.L)

    for x in [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1], [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
        print(dpp.log_prob(x).item())


if __name__ == '__main__':
    # test_dpp()
    test_backprop()
