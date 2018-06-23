"""Doing the real things in the code"""
#
import argparse
#
import numpy as np
import torch as th
import torch.nn.functional as F
#
from itertools import product
from math import ceil
from random import sample, shuffle
#
from numpy import isclose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions import Poisson
from tqdm import tqdm, trange
#
from differentiable_dpp import DeterminantalPointProcess, LEnsembleFactory
from inverse_nn import invert_our_diffeomorphism
from read_data import get_color_data
from sampler import AlignmentGibbsSampler
from trainer import PyTorchTrainer


DIM = 3
SCALAR_DIM = 1

MODEL = "dpp"


def write(*xs):
    tqdm.write(" ".join([str(x) for x in xs]))


def print(*xs):
    write(*xs)


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
        self.diffeomorphism[-1].weight.data = th.eye(DIM)
        # self.diffeomorphism[-1].weight.data = th.eye(DIM)

        self.λ = λ
        mus = self.init_prototypes(N)
        self.mus = nn.Parameter(mus)

    def init_prototypes(self, N):
        root = ceil(pow(N, 1/DIM))
        ticks = th.linspace(0.0, 20.0, root)
        points = list(product(ticks, repeat=DIM))
        points_used = sample(points, N)
        points_tensor = [th.tensor(p, requires_grad=True)
                         for p in points_used]
        altogether = th.stack(points_tensor)
        return altogether

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
        # print("Inventory: ", inventory)
        chromes = [color_to_chrome(color) for color in inventory]
        # print("Chromes: ", inventory)
        chromemes = list(μs[alignment])
        log_prob = th.tensor(0.)
        assert len(chromes) == len(chromemes)
        for chrome, chromeme in zip(chromes, chromemes):
            mvn = MultivariateNormal(chromeme, th.eye(DIM))
            result = mvn.log_prob(chrome)
            log_prob += result
        return log_prob

    def forward(self, training_data):
        step1 = self.step1_logprob(self.mus)
        step2 = self.step2_logprob(self.mus)
        step34 = th.tensor(0.0)

        LF = LEnsembleFactory(self.focalization_kernel)
        uses_focalization_only = MODEL == "bpp"
        L = LF.make(self.mus, use_dispersion=(not uses_focalization_only))
        dpp = DeterminantalPointProcess(L)
        color_to_chrome = invert_our_diffeomorphism(self.diffeomorphism)

        for language in training_data:
            inventory, alignments = language
            inventory = [th.Tensor(color) for color in inventory]
            alignment_logprobs = th.zeros(len(alignments))
            for i, alignment in enumerate(alignments):
                assert isinstance(alignment, list)
                alignment_logprobs[i] = (self.step3_logprob(self.mus, alignment, dpp) +
                                         self.step4_logprob(self.mus, alignment, inventory, color_to_chrome))
            step34 += logsumexp(alignment_logprobs, dim=0)
        return [-(step1 + step2 + step34)]  # Negative log-likelihood

    def cross_entropy(self, dev_data):
        """Cross-entropy of dev data.

        This function does *not* track gradients.
        """
        LF = LEnsembleFactory(self.focalization_kernel)
        uses_focalization_only = MODEL == "bpp"
        L = LF.make(self.mus, use_dispersion=(not uses_focalization_only))
        dpp = DeterminantalPointProcess(L)
        color_to_chrome = invert_our_diffeomorphism(self.diffeomorphism)

        step34 = 0.0
        for language in dev_data:
            inventory, alignments = language
            inventory = [th.Tensor(color) for color in inventory]
            alignment_logprobs = th.zeros(len(alignments))
            for i, alignment in enumerate(alignments):
                assert isinstance(alignment, list)
                alignment_logprobs[i] = (self.step3_logprob(self.mus, alignment, dpp) +
                                         self.step4_logprob(self.mus, alignment, inventory, color_to_chrome))
            step34 += logsumexp(alignment_logprobs, dim=0).detach().item()
        assert isinstance(step34, float)
        return -step34



def train_dev_test(data):
    length = len(data)
    split1, split2 = int(0.75 * length), int(0.875 * length)
    train, dev, test = data[:split1], data[split1:split2], data[split2:]
    return train, dev, test

def fit(whitener, inventories):
    triples = np.concatenate(inventories)
    whitener.fit(triples)

def transform(whitener, inventories):
    triples = np.concatenate(inventories)
    triples_whitened = list(whitener.transform(triples) / 1000)

    # Merge triples back into inventories.
    i = 0
    for idx, inventory in enumerate(inventories[:]):
        j = i + len(inventory)
        inventory_whitened = triples_whitened[i:j]
        assert len(inventory_whitened) == j - i, (len(inventory_whitened), j - i)
        inventories[idx] = inventory_whitened
        i = j
    assert j == len(triples_whitened)
    return inventories

def whiten(data):
    train, dev, test = data
    whitener = Pipeline([
        ('scl', StandardScaler()),
        ('wht', PCA(whiten=True))
    ])
    fit(whitener, train)
    transform(whitener, train)
    transform(whitener, dev)
    transform(whitener, test)

def prepare_training_data(N):
    color_foci = get_color_data()
    one_speaker_only = [speakers[0] for speakers in color_foci]
    # one_speaker_only = [inventory
    #                     for language in color_foci
    #                     for inventory in language]
    shuffle(one_speaker_only)
    train, dev, test = train_dev_test(one_speaker_only)
    whiten((train, dev, test))
    return train, dev, test


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    # If user doesn't specify an input file, read from standard input. Since
    # encodings are the worst thing, we're explicitly expecting std
    parser.add_argument("-m", "--model",
                        default="dpp")
    parser.add_argument("-N", "--num_prototypes",
                        type=int, default=50)
    return parser.parse_args()


def evaluate(model, train_data, dev_data):
    write("\tavg. X-ent train: ", model.cross_entropy(train_data) / len(train_data))
    write("\tavg. X-ent dev:   ", model.cross_entropy(dev_data) / len(dev_data))


def main():
    global MODEL
    args = parse_args()
    assert args.model in {"dpp", "bpp"}
    MODEL = args.model
    N = args.num_prototypes
    λ = 100
    data_train, data_dev, data_test = prepare_training_data(N)  # prepare_m_step_data(N)
    data_train = data_train  # CHANGE
    data_dev = data_dev  # CHANGE
    model = CompleteModel(λ=λ, N=N)
    n_iters = 10
    n_samples = 10

    prototypes = model.mus.detach().numpy()
    inverted = invert_our_diffeomorphism(model.diffeomorphism)

    samplers_train = [AlignmentGibbsSampler(prototypes, inventory, inverted) for inventory in data_train]  # CHANGE
    samplers_dev = [AlignmentGibbsSampler(prototypes, inventory, inverted) for inventory in data_dev]  # CHANGE

    # print(list(model.named_parameters()))

    for i in trange(n_iters, desc="EM round"):
        # E-step
        if i == i:  # CHANGE
            write(f"E-step {i}")
            burn_in = 100 if i == 0 else 0
            alignments_train = []
            for sampler in tqdm(samplers_train, desc="Language"):
                alignments_train.append([])
                for state in sampler.sample(n_samples, take_every_nth=20, burn_in=burn_in):
                    alignments_train[-1].append(list(state))
    
            alignments_dev = []
            for sampler in samplers_dev:
                alignments_dev.append([])
                for state in sampler.sample(n_samples, take_every_nth=20, burn_in=burn_in):
                    alignments_dev[-1].append(list(state))

        # M-step
        write(f"M-step {i}")
        assert len(data_train) == len(alignments_train)
        assert len(data_dev) == len(alignments_dev)
        train = list(zip(data_train, alignments_train))
        dev = list(zip(data_dev, alignments_dev))

        def eval_fn(model):
            return evaluate(model, train, dev)
        trainer = PyTorchTrainer(model, epochs=4, evaluate=eval_fn)
        trainer.train(train)

        # update sampler's prototypes
        for sampler in samplers_train + samplers_dev:
            sampler.chromemes = model.mus.detach()


if __name__ == '__main__':
    main()
