"""
INCOMPLETE. DON'T USE ME YET.
"""

from collections import Counter, defaultdict
import csv
from pprint import pprint
from random import choice
from functools import lru_cache
from itertools import product

import pandas as pd

from numpy import pi as π
import torch as th
from torch.autograd import Variable
from torch import nn
from torch.distributions import Normal
from torchtext.vocab import Vocab
from scipy.stats import poisson


class BigModel(th.nn.modules.Module):
    """docstring for BigModel"""
    def __init__(self, chips: Vocab, EMBED_DIM=40):
        super(BigModel, self).__init__()
        self.embeds1 = th.nn.Linear(3, EMBED_DIM)
        self.embeds2 = th.nn.Linear(EMBED_DIM, EMBED_DIM // 2)
        self.λ = 100

        self.dim_of_space = 3

        focalizer = nn.Sequential(
            nn.Linear(self.dim_of_space, self.dim_of_space),
            # nn.Tanh(),
            nn.Linear(self.dim_of_space, 1)
            )

    def step_1(self):
        N = poisson.rvs(mu=self.λ)
        log_likelihood = poisson.logpmf(N, mu=self.λ)
        return N, log_likelihood

    def step_2(self, N):
        d = self.dim_of_space
        μ̃ = th.Tensor([0])
        eye = th.Tensor([1])
        normal = Normal(μ̃, eye)
        samples = [th.stack([normal.sample()
                             for _
                             in range(d)
                             ])
                   for __
                   in range(N)
                   ]
        log_likelihood = sum(normal.log_prob(row).sum() for row in samples)
        return samples, log_likelihood

    def kernel(self, μ, μʹ, ρ):
        d = self.dim_of_space
        σ = 1

        term1 = (2 * ρ) ** (d / 2)
        term2 = (2 * π * σ**2) ** ((1 - 2*ρ) * d / 2)
        term3 = th.exp(- (ρ) * (th.norm(μ - μʹ) ** 2) / (4 * σ**2))
        return term1 * term2 * term3

    def focalization(self, μ):
        return th.exp(th.tanh())

    def define_DPP(self, μs, ρ=None):
        if not ρ:
            ρ = th.Tensor([0.01])
        N = len(μs)
        L = th.zeros(N, N)
        for (i, μ), (j, μʹ) in product(enumerate(μs), 2):
            L[i, j] = self.kernel(μ, μʹ, ρ)
        for i, μ in enumerate(μs):
            L[i, j] += self.focalization(μ)
        return L

    def forward(self, language_colors: set):
        N, log_likelihood1 = self.step_1()
        μ, log_likelihood2 = self.step_2(N)
        rgb_colors = [Variable(th.Tensor(wcs_to_rgb(color)))
                      for color in language_colors]
        embeds = [self.embeds1(color) for color in rgb_colors]
        embeds = [th.nn.functional.tanh(e) for e in embeds]
        embeds = [self.embeds2(e) for e in embeds]
        phis = [e.norm() for e in embeds]
        log_phis = [phi.log() for phi in phis]
        # w = self.w[language_colors.data].squeeze(1)
        # t = th.nn.functional.softmax(w, dim=0)
        log_probability = Variable(th.Tensor([0]))
        for log_phi in log_phis:
            log_probability += log_phi
        return log_probability


def train_model(model, iterator, *, n_iters=50, learning_rate=1e-1):
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(n_iters):
        corpus_neg_log_loss = Variable(th.Tensor([0]))
        for batch in iterator:
            neg_log_loss = -1 * model(batch)
            corpus_neg_log_loss += neg_log_loss
        print(corpus_neg_log_loss.data[0])
        optimizer.zero_grad()
        corpus_neg_log_loss.backward()
        optimizer.step()


@lru_cache()
def wcs_to_rgb_database(filename="data/cnum-vhcm-lab-new.txt"):
    df = pd.read_table("data/cnum-vhcm-lab-new.txt")
    df["wcs"] = df["V"] + df["H"].astype(str)
    df.set_index("wcs", inplace=True)
    df = df[["L*", "a*", "b*"]]
    return df.loc


@lru_cache(maxsize=512)
def wcs_to_rgb(chip: str):
    database = wcs_to_rgb_database()
    return database[chip]


def read_colors():
    terms_for_language = defaultdict(set)
    with open("data/term.txt") as f:
        reader = csv.DictReader(f, delimiter="\t",
                                fieldnames=["Language",
                                            "Speaker",
                                            "Chip",
                                            "Term"])
        for row in reader:
            terms_for_language[row["Language"]].add(row["Term"])

    pprint(terms_for_language)
    print(Counter(len(x) for x in terms_for_language.values()))


def valid_colors(iterable):
    for row in iterable:
        if row["Chip"].startswith("A") and row["Chip"] != "A0":
            continue
        if row["Chip"].startswith("J") and row["Chip"] != "J0":
            continue
        yield row


def main():
    chips_for_language = defaultdict(list)
    with open("data/foci-exp.txt") as f:
        reader = csv.DictReader(f, delimiter="\t", 
                                fieldnames=["Language",
                                            "Speaker",
                                            "Response",
                                            "Abbrev",
                                            "Chip"])
        for row in valid_colors(reader):
            if row["Speaker"] == "1":  # Just look at one speaker.
                chips_for_language[row["Language"],
                                   row["Response"]
                                   ].append(row["Chip"])
    chips_reduced = {k: choice(v) for k, v in chips_for_language.items()}
    # pprint(chips_reduced)
    vocabs_for_each_language = defaultdict(set)
    for (lang, idx), chip in chips_reduced.items():
        vocabs_for_each_language[lang].add(chip)

    just_chipsets = [chips for lang, chips in vocabs_for_each_language.items()]

    counter = Counter()
    for (_, chips) in vocabs_for_each_language.items():
        for chip in chips:
            counter[chip] += 1
    vocab = Vocab(counter)
    print(vocab.stoi)

    model = BigModel(vocab)

    for chipset in just_chipsets:
        print([vocab.stoi[x] for x in chipset])

    train_model(model, just_chipsets)


if __name__ == '__main__':
    main()
