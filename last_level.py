import numpy as np
import torch as th

from scipy.stats import multivariate_normal as mvn
from torch import nn
from torch.autograd import Variable


from inverse_nn import invert_our_diffeomorphism

th.random.manual_seed(1337)
np.random.seed(1337)



class ChromemesToColors(th.nn.modules.Module):
    """docstring for ChromemesToColors"""
    def __init__(self, dim):
        super(ChromemesToColors, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(3, 3),
            nn.Tanh(),
            nn.Linear(3, 3),
        )

    @staticmethod
    def loss(input_, target):
        return th.sum((input_ - target) ** 2)

    def forward(self, prototypes, manifestations, alignment):
        colors = manifestations
        chromemes = prototypes

        total_loss = Variable(th.Tensor([0]))
        for i, color in enumerate(colors):
            color = Variable(th.from_numpy(color).float(), requires_grad=False)
            corresponding_chromeme = chromemes[alignment[i]]
            chrome = mvn.rvs(corresponding_chromeme,
                             np.eye(len(corresponding_chromeme)))
            chrome = Variable(th.from_numpy(chrome).float())
            color_pred = self.mlp(chrome)
            total_loss += self.loss(color, color_pred)
        return total_loss


def train_model(model, prot, manif, alig, *, n_iters=10, learning_rate=1e-2):
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(n_iters):
        corpus_loss = Variable(th.Tensor([0]))
        for al in alig:
            loss = model(prot, manif, al)
            corpus_loss += loss
        print(corpus_loss.data[0])
        optimizer.zero_grad()
        corpus_loss.backward()
        optimizer.step()


def assert_invertible(tensor):
    assert not np.isclose(np.linalg.det(tensor), 0)


if __name__ == '__main__':
    def mvn_rv(dim=3):
        return mvn.rvs(np.zeros(dim))

    spatial_dim = 3
    n_prototypes = 100
    n_manifestations = 20

    prototypes = [mvn_rv(spatial_dim) for _ in range(n_prototypes)]
    manifestations = [mvn_rv(spatial_dim) for _ in range(n_manifestations)]

    alignments = [[66, 98, 71, 83, 74, 91, 34, 15, 39, 8, 78, 77, 25, 54, 27, 95, 57, 13, 41, 73],
                  [66, 96, 71, 83, 74, 91, 35, 38, 37, 59, 21, 77, 69, 54, 24, 95, 79, 39, 41, 56],
                  [17, 19, 14, 0, 5, 91, 35, 60, 62, 59, 87, 77, 69, 72, 88, 32, 33, 55, 41, 82],
                  [17, 19, 14, 0, 5, 53, 81, 60, 62, 59, 87, 77, 65, 2, 88, 39, 33, 55, 41, 13],
                  [45, 19, 14, 0, 78, 9, 49, 60, 72, 59, 87, 77, 65, 2, 16, 24, 27, 92, 41, 13],
                  [17, 19, 4, 46, 40, 91, 49, 69, 72, 67, 32, 77, 20, 2, 16, 24, 35, 92, 52, 13],
                  [73, 71, 4, 28, 40, 98, 49, 69, 50, 67, 48, 77, 36, 2, 16, 54, 47, 92, 52, 13],
                  [73, 71, 96, 28, 9, 98, 91, 69, 15, 44, 49, 65, 36, 30, 17, 27, 76, 7, 31, 13],
                  [40, 71, 14, 73, 63, 98, 97, 69, 70, 51, 49, 3, 39, 15, 17, 37, 25, 7, 31, 13],
                  [40, 0, 14, 73, 63, 98, 54, 52, 70, 51, 49, 8, 39, 85, 93, 37, 17, 7, 55, 13],
                  [40, 0, 9, 73, 76, 82, 54, 96, 77, 89, 47, 20, 23, 84, 93, 37, 86, 7, 55, 61],
                  [10, 80, 81, 73, 76, 82, 6, 90, 77, 89, 2, 20, 23, 84, 70, 37, 35, 63, 55, 61],
                  [10, 25, 48, 73, 76, 21, 6, 90, 77, 45, 0, 88, 23, 8, 2, 27, 44, 50, 14, 61],
                  [10, 83, 48, 5, 26, 21, 68, 46, 77, 45, 49, 57, 23, 8, 2, 11, 28, 50, 91, 52],
                  [10, 83, 48, 5, 26, 21, 68, 97, 4, 32, 49, 41, 23, 8, 2, 11, 28, 50, 91, 52],
                  [10, 83, 33, 5, 85, 21, 68, 46, 38, 95, 49, 41, 1, 26, 2, 11, 32, 9, 59, 52],
                  [81, 45, 33, 37, 86, 28, 68, 94, 38, 95, 88, 60, 1, 87, 69, 11, 32, 18, 59, 52],
                  [81, 56, 33, 37, 10, 28, 9, 99, 82, 95, 42, 60, 1, 76, 69, 49, 32, 94, 59, 35],
                  [73, 0, 33, 65, 2, 28, 9, 99, 58, 71, 10, 60, 1, 76, 7, 26, 36, 88, 83, 8],
                  [38, 86, 32, 20, 2, 28, 9, 34, 58, 71, 81, 60, 1, 76, 31, 26, 36, 88, 74, 8],
                  [38, 86, 92, 56, 80, 28, 9, 34, 58, 71, 21, 60, 1, 76, 51, 26, 84, 12, 74, 8],
                  [87, 86, 36, 56, 43, 23, 9, 34, 62, 61, 21, 66, 50, 39, 51, 26, 63, 12, 74, 65],
                  [87, 1, 36, 13, 15, 69, 9, 83, 62, 81, 22, 16, 50, 39, 60, 26, 42, 48, 8, 17],
                  [0, 51, 36, 13, 15, 3, 57, 37, 86, 4, 22, 75, 50, 34, 60, 41, 42, 48, 8, 17],
                  [0, 51, 36, 13, 15, 62, 57, 33, 86, 89, 74, 75, 50, 34, 60, 41, 70, 48, 96, 72],
                  [80, 18, 36, 13, 15, 81, 57, 19, 68, 89, 74, 75, 50, 34, 6, 41, 99, 48, 96, 72],
                  [91, 3, 59, 13, 8, 71, 25, 83, 68, 89, 74, 9, 50, 42, 12, 41, 65, 60, 57, 95],
                  [37, 3, 59, 13, 8, 71, 11, 83, 39, 98, 74, 91, 27, 42, 9, 41, 65, 76, 57, 52],
                  [37, 45, 81, 13, 36, 71, 11, 83, 39, 67, 2, 72, 16, 98, 9, 41, 89, 76, 57, 99],
                  [37, 45, 28, 13, 36, 78, 11, 29, 84, 67, 20, 58, 16, 98, 55, 41, 47, 76, 71, 2],
                  [37, 97, 86, 63, 64, 95, 69, 29, 84, 39, 20, 57, 16, 44, 55, 40, 98, 76, 65, 2],
                  [80, 75, 24, 19, 64, 77, 12, 3, 61, 39, 20, 57, 97, 44, 55, 40, 45, 22, 18, 49],
                  [78, 83, 62, 19, 64, 77, 85, 3, 61, 32, 43, 57, 29, 31, 0, 20, 66, 22, 18, 99],
                  [78, 83, 62, 81, 90, 77, 85, 50, 40, 63, 43, 57, 10, 31, 0, 96, 32, 22, 14, 64],
                  [78, 34, 62, 81, 67, 94, 85, 7, 40, 75, 43, 73, 10, 31, 28, 69, 32, 47, 14, 49],
                  [78, 5, 84, 81, 67, 35, 85, 7, 40, 0, 62, 74, 10, 9, 34, 69, 2, 47, 21, 49],
                  [78, 5, 84, 50, 67, 35, 97, 7, 40, 83, 59, 74, 10, 76, 34, 69, 2, 47, 21, 24],
                  [78, 64, 0, 28, 67, 13, 97, 56, 40, 83, 79, 50, 10, 11, 95, 69, 22, 71, 21, 84],
                  [52, 64, 41, 28, 67, 73, 97, 20, 39, 83, 77, 76, 91, 11, 95, 55, 22, 71, 7, 89],
                  [18, 44, 41, 28, 67, 85, 97, 36, 64, 83, 77, 76, 91, 11, 95, 55, 22, 1, 7, 89],
                  [18, 44, 41, 10, 67, 17, 97, 66, 59, 83, 77, 73, 91, 51, 93, 55, 56, 1, 7, 89],
                  [18, 60, 72, 10, 32, 17, 16, 97, 59, 83, 77, 12, 91, 58, 93, 55, 54, 1, 7, 89],
                  [18, 20, 72, 78, 10, 94, 80, 99, 59, 29, 77, 12, 91, 68, 47, 8, 32, 85, 27, 90],
                  [26, 20, 72, 78, 2, 63, 38, 99, 37, 36, 77, 80, 4, 68, 47, 8, 32, 85, 76, 90],
                  [26, 20, 72, 33, 2, 63, 44, 99, 37, 96, 22, 43, 4, 65, 47, 8, 49, 85, 19, 81],
                  [61, 1, 41, 27, 2, 63, 44, 99, 37, 38, 23, 43, 4, 18, 87, 93, 49, 68, 40, 83],
                  [61, 1, 41, 36, 2, 63, 44, 99, 37, 38, 23, 78, 4, 85, 16, 93, 49, 11, 34, 83],
                  [72, 90, 29, 36, 2, 63, 17, 99, 37, 47, 23, 91, 4, 85, 16, 9, 49, 11, 19, 42],
                  [72, 24, 0, 48, 43, 75, 10, 60, 67, 26, 23, 15, 4, 85, 16, 28, 98, 11, 19, 42],
                  [83, 24, 0, 48, 7, 29, 34, 60, 67, 80, 57, 38, 4, 85, 78, 28, 98, 59, 19, 11],
                 ]
    model = ChromemesToColors(dim=spatial_dim)
    train_model(model, prototypes, manifestations, alignments)
    assert_invertible(model.mlp[0].weight.data)
    assert_invertible(model.mlp[-1].weight.data)
    print(model.mlp[0].weight.data)
    print(model.mlp[-1].weight.data)
