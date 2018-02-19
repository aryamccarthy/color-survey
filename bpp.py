from collections import Counter, defaultdict
import csv
from pprint import pprint
from random import choice
from functools import lru_cache

import pandas as pd

import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchtext.vocab import Vocab
from torchtext import data
from torchtext.vocab import Vocab


class BernoulliPointProcess(th.nn.modules.Module):
    """docstring for BernoulliPointProcess"""
    def __init__(self, chips: Vocab, EMBED_DIM=40):
        super(BernoulliPointProcess, self).__init__()
        self.embeds1 = th.nn.Linear(3, EMBED_DIM)
        self.embeds2 = th.nn.Linear(EMBED_DIM, EMBED_DIM // 2)
        # self.w = Parameter(th.zeros(len(chips), EMBED_DIM))
        # th.nn.init.normal(self.w, std=1/(6 ** (1/2)))

    def forward(self, language_colors: set):
        rgb_colors = [Variable(th.Tensor(munsell_to_rgb(color))) for color in language_colors]
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


@lru_cache(maxsize=512)
def munsell_to_rgb(chip: str):
    with open("conversion.txt") as f:
        for line in f:
            line = line.strip().split()
            if line[0] == chip:
                return [int(x) for x in line[1:]]
        else:
            raise KeyError(chip)


def read_colors():
    terms_for_language = defaultdict(set)
    with open("data/term.txt") as f:
        reader = csv.DictReader(f, delimiter="\t",
                                fieldnames=["Language", "Speaker", "Chip", "Term"])
        for row in reader:
            terms_for_language[row["Language"]].add(row["Term"])

    pprint(terms_for_language)
    print(Counter(len(x) for x in terms_for_language.values()))


def main():
    chips_for_language = defaultdict(list)
    with open("data/foci-exp.txt") as f:
        reader = csv.DictReader(f, delimiter="\t", 
                                fieldnames=["Language",
                                            "Speaker",
                                            "Response",
                                            "Abbrev",
                                            "Chip"])
        for row in reader:
            if row["Chip"].startswith("A") and row["Chip"] != "A0":
                continue
            if row["Chip"].startswith("J") and row["Chip"] != "J0":
                continue
            if row["Speaker"] == "1":  # Just look at one speaker.
                chips_for_language[row["Language"], row["Response"]].append(row["Chip"])
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

    model = BernoulliPointProcess(vocab)

    for chipset in just_chipsets:
        print([vocab.stoi[x] for x in chipset])
    # indexes = [Variable(th.LongTensor([vocab.stoi[chip]
    #                           for x
    #                           in chipset
    #                           ]
    #                          ), requires_grad=False)
    #            for chipset
    #            in just_chipsets
    #            ]

    train_model(model, just_chipsets)
    # df = pd.read_table("data/term.txt", header=None,
    #                    names=["Language", "Speaker", "Chip", "Term"],
    #                    index_col=[0, 1, 2])
    # print(df.head())


if __name__ == '__main__':
    main()
