import numpy as np
import torch as th
import torch.optim as optim

from itertools import chain, combinations

from numpy.random import uniform
from scipy.linalg import orth
from torch.nn import Parameter
from torch.nn.modules import Module


class DeterminantalPointProcess(Module):
    """docstring for DeterminantalPointProcess"""
    def __init__(self, L):
        super(DeterminantalPointProcess, self).__init__()
        assert (L == L.t()).all(), "L must be symmetric!"
        self._log_L = Parameter(th.log(L))
        self.N = L.shape[0]
        # w, v = th.symeig(self.L, eigenvectors=True)
        # assert (w > 0).all(), "L must be positive definite."
        # self.w, self.v = w, v

    @property
    def L(self):
        return th.exp(self._log_L)

    @property
    def w(self):
        """Eigenvalues"""
        w, v = th.symeig(self.L, eigenvectors=True)
        assert (w > 0).all(), "L must be positive definite."
        return w

    @property
    def v(self):
        """Eigenvectors"""
        w, v = th.symeig(self.L, eigenvectors=True)
        assert (w > 0).all(), "L must be positive definite."
        return v

    @property
    def log_normalizer(self):
        w, v = th.symeig(self.L, eigenvectors=True)
        assert (w > 0).all(), "L must be positive definite."
        
        log_normalizer = th.logdet(self.L + th.eye(self.N))
        return log_normalizer

    @property
    def support(self):
        def powerset(iterable):
            xs = list(iterable)
            return chain.from_iterable(combinations(xs, n)
                                       for n in range(len(xs) + 1))
        items = range(self.N)
        support = list(th.LongTensor(x) for x in powerset(items))
        return support

    def supported(self, x):
        for v in self.support:
            if v.shape == x.shape and (v == x).all():
                return True
        else:
            return False

    def log_prob(self, x):
        assert self.supported(x), f"You can't draw that from this DPP: {x}"
        if len(x) == 0:
            logdet = th.log(th.Tensor([1]))
        else:
            coordinates = th.LongTensor(x)
            submatrix = self.L[coordinates][:, coordinates]
            print("Submatrix: {}".format(submatrix.data.numpy()))
            logdet = th.logdet(submatrix)
            print("Logdet: {}".format(logdet.data))
        return logdet - self.log_normalizer

    def sample(self):
        # 1. Select elementary DPP
        probs = (self.w / (self.w + 1)).detach().numpy()
        index = uniform(size=self.N) <= probs
        V = self.v.detach().numpy()[:, index]

        # 2. Draw sample from selected elementary DPP
        k = np.sum(index)
        J = []
        for i in range(k):
            p = np.mean(V ** 2, axis=1)  # element-wise square
            p = np.cumsum(p)
            item = (np.random.uniform() <= p).argmax()
            J.append(item)

            # Delete one eigenvector not orthogonal to e_item,
            # then find a new basis.
            j = (np.abs(V[item, :]) > 0).argmax()
            Vj = V[:, j]
            V = orth(V - (np.outer(Vj, (V[item, :] / Vj[item]))))

        J.sort()
        sample = th.LongTensor(J)
        return sample

    def forward(self, x):
        _ = self.v
        return self.log_prob(x)


def test_dpp():
    M = th.Tensor([[2,    0.25,    0.5],
                   [0.25,    2,   0.25],
                   [0.5,  0.25,      2]])
    dpp = DeterminantalPointProcess(M)

    # Test validity as probability distribution.
    total = 0.0
    for elem in dpp.support:
        assert dpp.supported(elem), "Shoot! It's not supported."
        total += np.exp(dpp.log_prob(elem).data)
    assert np.isclose(total, 1.0), "Not a valid probability distribution!"

    # Test sampling and log_prob
    for i in range(100):
        sample = dpp.sample()
        prob = dpp.log_prob(sample)


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
    data = [th.LongTensor(t) for t in data]

    M = th.Tensor([[2,    0.25,    0.5],
                   [0.25,    2,   0.25],
                   [0.5,  0.25,      2]])
    # M = th.Tensor([[2,     -1,      0],
    #                [-1,     2,     -1],
    #                [0,     -1,    2]])
    dpp = DeterminantalPointProcess(M)

    optimizer = optim.SGD(dpp.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(20):
        loss = th.Tensor([0])
        optimizer.zero_grad()
        for datum in data:
            log_prob = dpp(datum)
            print(f"\t{float(log_prob)}")
            loss -= log_prob
        print(dpp.L)
        for x in dpp.support:
            print(f"{x.data.numpy()}\t{dpp.log_prob(x).data[0]}")
        print(float(loss.data) / len(data))
        print("X" * int(loss.data))
        loss.backward()
        optimizer.step()
    print(dpp.L)


if __name__ == '__main__':
    test_dpp()
    test_backprop()
