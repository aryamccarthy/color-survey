import torch as th
#
from math import factorial, log
from random import sample
#
from numpy import isclose
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions import Poisson
from tqdm import tqdm, trange
#
from differentiable_dpp import DeterminantalPointProcess, LEnsembleFactory
from inverse_nn import invert_our_diffeomorphism
from read_data import get_color_data
from trainer import PyTorchTrainer


DIM = 3
SCALAR_DIM = 1



import torch.nn.functional as F

def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs, dim=0)).mean(dim, keepdim=keepdim)


class CompleteModel(nn.Module):
    def __init__(self, λ, N):
        super(CompleteModel, self).__init__()
        self.focalization_kernel = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.Tanh(),
            nn.Linear(DIM, SCALAR_DIM)
        )
        self.diffeomorphism = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.Tanh(),
            nn.Linear(DIM, DIM)
        )

        self.λ = λ
        self.mus = nn.Parameter(th.randn(N, DIM))
        print(self.diffeomorphism[0].weight)

    def step1_logprob(self, μs):
        N = len(μs)
        poisson = Poisson(self.λ)
        return poisson.log_prob(N)

    def step2_logprob(self, μs):
        mean = th.zeros(DIM)
        covariance = th.eye(DIM)
        mvn = MultivariateNormal(mean, covariance)
        result = mvn.log_prob(μs).sum()
        return result

    def step3_logprob(self, μs, alignment, dpp):
        assert isclose(dpp.log_prob(alignment).item(), dpp.log_prob(alignment[::-1]).item())
        return dpp.log_prob(alignment)

    def step4_logprob(self, μs, alignment, inventory, color_to_chrome):
        #print(inventory)
        chromes = [color_to_chrome(color) for color in inventory]
        #print(chromes)
        chromemes = list(μs[alignment])
        log_prob = th.tensor(0.)
        assert len(chromes) == len(chromemes)
        for chrome, chromeme in zip(chromes, chromemes):
            mvn = MultivariateNormal(chromeme, th.eye(DIM))
            result = mvn.log_prob(chrome)
            # print(chrome.detach().numpy(), chromeme.detach().numpy(), result)
            log_prob += result
        return log_prob

    def forward(self, all_data):
        color_foci, all_alignments = all_data
        step1 = self.step1_logprob(self.mus)
        step2 = self.step2_logprob(self.mus)
        step34 = th.tensor(0.0)

        print(th.det(self.diffeomorphism[0].weight))
        print(th.det(self.diffeomorphism[-1].weight))

        L = LEnsembleFactory(self.focalization_kernel).make(self.mus)
        dpp = DeterminantalPointProcess(L)
        color_to_chrome = invert_our_diffeomorphism(self.diffeomorphism)

        assert len(color_foci) == len(all_alignments)
        for language in zip(color_foci, all_alignments):
            inventory, alignments = language
            inventory = [th.Tensor(color) for color in inventory]
            alignment_logprobs = th.zeros(len(alignments))
            for i, alignment in enumerate(alignments):
                assert isinstance(alignment, list)
                alignment_logprobs[i] = (self.step3_logprob(self.mus, alignment, dpp) +
                                         self.step4_logprob(self.mus, alignment, inventory, color_to_chrome))
            step34 += logsumexp(alignment_logprobs, dim=0)
        return [-(step1 + step2 + step34)]


def perform_expectation_step(model, n_samples=5):
    pass


def initial_alignment(N, n_l):
    return sample(range(N), n_l)


def prepare_training_data(N):
    color_foci = get_color_data()
    one_speaker_only = [speakers[0] for speakers in color_foci]
    alignments = [initial_alignment(N, len(inventory)) for inventory in one_speaker_only]
    return (one_speaker_only, alignments)


def prepare_m_step_data(N):
    color_foci = get_color_data()
    one_speaker_only = [speakers[0] for speakers in color_foci]
    alignments = [[initial_alignment(N, len(inventory)) for _ in range(5)] for inventory in one_speaker_only]
    return (one_speaker_only, alignments)


def main():
    N = 50
    λ = 100
    data = prepare_m_step_data(N)
    print(data[1][1])
    model = CompleteModel(λ=λ, N=N)
    n_iters = 5
    n_samples = 10

    def write(x):
        tqdm.write(str(x))

    for i in trange(n_iters, desc="EM round"):
        # alignments = perform_expectation_step(model, n_samples)
        trainer = PyTorchTrainer(model, epochs=4)
        # write(model.mus[0])
        write(model.diffeomorphism[0].weight)
        write(model.diffeomorphism[0].bias)
        write(model.diffeomorphism[-1].weight)
        write(model.diffeomorphism[-1].bias)
        trainer.train(data)





if __name__ == '__main__':
    main()