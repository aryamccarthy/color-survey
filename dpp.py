from itertools import chain, combinations

import numpy as np
from numpy.random import uniform
from scipy.linalg import orth


def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


class DeterminantalPointProcess(object):
    """docstring for DeterminantalPointProcess

    Loosely based on
    github.com/mbp28/determinantal-point-processes/blob/master/sampling/sample_dpp.py
    """
    def __init__(self, L):
        super(DeterminantalPointProcess, self).__init__()
        assert (L == L.transpose()).all(), "L must be symmetric!"
        self.L = L
        self.N = L.shape[0]
        w, v = np.linalg.eigh(self.L)
        assert (w > 0).all(), "L must be positive definite."
        self.w, self.v = w, v

    @lazy_property
    def log_normalizer(self):
        _, log_normalizer = np.linalg.slogdet(self.L + np.eye(self.N))
        return log_normalizer

    @lazy_property
    def support(self):
        def powerset(iterable):
            """
            powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
            """
            xs = list(iterable)
            # note we return an iterator rather than a list
            return chain.from_iterable(combinations(xs, n)
                                       for n in range(len(xs)+1))
        items = range(self.N)
        support = list(np.array(x) for x in powerset(items))
        return support

    def supported(self, x):
        for v in self.support:
            if v.shape == x.shape and (v == x).all():
                return True
        else:
            return False

    def sample(self):
        # 1. Select elementary DPP
        probs = self.w / (self.w + 1)
        index = uniform(size=self.N) <= probs
        V = self.v[:, index]

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
        sample = np.asarray(J)
        return sample

    def log_prob(self, x):
        assert self.supported(x), f"You can't draw that from this DPP: {x}"
        submatrix = self.L[np.ix_(x, x)]  # Get the right submatrix.
        _, logdet = np.linalg.slogdet(submatrix)

        return logdet - self.log_normalizer


class TestDeterminantalPointProcess(object):
    """docstring for TestDeterminantalPointProcess"""
    def __init__(self):
        super(TestDeterminantalPointProcess, self).__init__()

    def run_tests(self):
        M = np.array([[2,   -1,   0],
                      [-1,   2,  -1],
                      [0,   -1, 100]])
        dpp = DeterminantalPointProcess(M)
        # Test that probs work.
        for _ in range(100):
            sample = dpp.sample()
            dpp.log_prob(sample)

        assert np.isclose(sum([np.exp(dpp.log_prob(x))
                               for x in dpp.support]),
                          1
                          )


if __name__ == '__main__':
    TestDeterminantalPointProcess().run_tests()
