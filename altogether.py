#
import warnings

#
import pandas as pd
import torch as th

#
from torch import nn
from torch.autograd import Variable
from torch.distributions import Poisson, MultivariateNormal
from torch.utils import trainer
from torch.utils.trainer.plugins import ProgressMonitor, LossMonitor, Logger

#
from first_two_steps import LEnsembleFactory
from read_inputs import produce_color_lists

th.manual_seed(137)


def Scalar(x):
    assert type(x) in {int, float}
    return th.Tensor([x])


class Altogether(nn.modules.Module):
    def __init__(self, λ, dim):
        super(Altogether, self).__init__()
        self.λ = λ
        self.dim = dim

        self.poisson = Poisson(λ)
        self.steps_one_and_two()

        self.LEnsembleFactory = LEnsembleFactory(self.dim)

    def steps_one_and_two(self):
        self.N = int(self.poisson.sample())
        μs = th.Tensor(self.N, self.dim)
        self.μs = nn.Parameter(μs, requires_grad=True)
        nn.init.normal_(self.μs)

    def step1_logprob(self):
        return_ = self.poisson.log_prob(self.N)
        return return_

    def step2_logprob(self):
        mean = th.zeros(self.dim)
        covariance = th.eye(self.dim)
        mvn = MultivariateNormal(mean, covariance)
        return_ = mvn.log_prob(self.μs).sum()
        return return_

    def forward(self, all_data):
        return self.step1_logprob() + self.step2_logprob()


def train(model, train_loader):
    optimizer = th.optim.Adam(model.parameters(), lr=0.01)

    def loss(batch_output, _):
        return -batch_output

    t = trainer.Trainer(model, loss, optimizer, train_loader)
    t.register_plugin(ProgressMonitor())
    t.register_plugin(LossMonitor())
    t.register_plugin(Logger(['progress', 'loss']))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', "invalid index of a 0-dim tensor", UserWarning)
        t.run(epochs=10)


def compute_expectation(model):
    pass


def make_data():
    # samples = range(100)
    # data = [(Scalar(x), None)
    #         for x in samples
    #         ]
    samples = [th.Tensor(inventory) for inventory in produce_color_lists()]
    data = [(x, None) for x in samples]
    return data


def main():
    λ = 100
    dim = 3
    model = Altogether(λ ,dim)

    data = make_data()
    print(data)

    n_em_iters = 1
    for i in range(n_em_iters):
        compute_expectation(model)
        train(model, data)
    # print(model.μs)


if __name__ == '__main__':
    main()