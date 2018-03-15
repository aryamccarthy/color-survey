

#
import heapq
import logging
import math

#
import torch as th

#
from pathlib import Path

#
from scipy.stats import poisson
from scipy.stats import multivariate_normal as mvn
from torch import nn
from torch.autograd import Variable
from torch.utils import trainer
from torch.utils.trainer.plugins import ProgressMonitor, LossMonitor, Logger


log = logging.getLogger(Path(__file__).stem)


def Scalar(x):
    assert type(x) in {int, float}
    return th.Tensor([x])


class StepOne(nn.modules.Module):
    """docstring for StepOne"""
    def __init__(self):
        super(StepOne, self).__init__()
        self.λ = nn.Parameter(Scalar(100))

    def log_likelihood(self, value):
        """Compute Poisson log likelihood."""
        rate = self.λ
        return (rate.log() * value) - rate - (value + 1).lgamma()

    def rv(self):
        return Scalar(int(poisson.rvs(self.λ.data)))

    def forward(self, value):
        return self.log_likelihood(value)


class StepTwo(nn.modules.Module):
    """docstring for StepTwo"""
    def __init__(self, dim):
        super(StepTwo, self).__init__()
        # self.dim = dim
        self.mvn_μ = nn.Parameter(th.zeros(dim))
        self.mvn_σ = Variable(th.eye(dim), requires_grad=False)
        for i in range(dim):  # Ensure no covariance.
            for j in range(i):
                assert self.mvn_σ[i, j].data[0] == 0
                assert self.mvn_σ[j, i].data[0] == 0

    @staticmethod
    def normal_log_likelihood(value, loc, scale):
        var = scale ** 2
        return (-((value - loc) ** 2) / (2 * var) -
                scale.log() -
                math.log(math.sqrt(2 * math.pi))
                )

    def log_likelihood(self, value):
        log_liks = self.normal_log_likelihood(value,
                                              self.mvn_μ,
                                              th.diag(self.mvn_σ)
                                              )
        return log_liks.sum()

    def rv(self):
        return mvn.rvs(mean=self.mvn_μ.data, cov=self.mvn_σ.data)

    def forward(self, inventory):
        log_liks = []
        for i in range(inventory.shape[1]):
            log_liks.append(self.log_likelihood(inventory[:, i]))
        return sum(log_liks)


class LEnsembleFactory(nn.modules.Module):
        """docstring for LEnsembleFactory"""
        def __init__(self, dim):
            super(LEnsembleFactory, self).__init__()
            self.dim = dim

            self.focalizer = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh(),
                nn.Linear(dim, 1),
                )

        def kernel(self, μ, μʹ, ρ):
            d = self.dim
            σ = 1

            term1 = (2 * ρ) ** (d / 2)
            term2 = (2 * π * σ**2) ** ((1 - 2*ρ) * d / 2)
            term3 = th.exp(- (ρ) * (th.norm(μ - μʹ) ** 2) / (4 * σ**2))
            return term1 * term2 * term3

        def focalization(self, μ):
            return th.exp(self.focalizer(μ))

        def make(self, μs, ρ=None):
            if not ρ:
                ρ = Scalar(0.01)
            N = len(μs)
            L = th.zeros(N, N)
            for (i, μ), (j, μʹ) in product(enumerate(μs), 2):
                L[i, j] = self.kernel(μ, μʹ, ρ)
            for i, μ in enumerate(μs):
                L[i, j] += self.focalization(μ)
            return L


class FirstTwoSteps(nn.modules.Module):
    def __init__(self, dim):
        super(FirstTwoSteps, self).__init__()
        self.dim = dim
        self.step1 = StepOne()
        self.step2 = StepTwo(dim)

    def forward(self, value):
        return self.step2(value) + self.step1(Variable(Scalar(value.shape[1])))


def train(model, train_loader):
    optimizer = th.optim.Adam(model.parameters(), lr=1)

    def loss(batch_output, _):
        return -batch_output

    t = trainer.Trainer(model, loss, optimizer, train_loader)
    t.register_plugin(ProgressMonitor())
    t.register_plugin(LossMonitor())
    # t.register_plugin(Logger(['progress', 'loss']))

    t.run(epochs=3)


def main():
    model = FirstTwoSteps(dim=3)

    # Generation
    N = int(model.step1.rv()[0])

    μs = [model.step2.rv() for _ in range(N)]
    print(μs)

    # TODO: sample DPP

    # Testing backpropagation.
    counts = poisson.rvs(mu=20, size=80)
    samples = [th.rand(3, int(count))
               for count in counts]
    data = [(x, None)
            for x in samples
            ]
    train(model, data)
    print(model.step1.λ.data)
    print(model.step2.mvn_μ.data)


if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    main()
