import torch as th

from numpy import isclose
from torch import nn

th.manual_seed(1337)


def inv_tanh(x):
    return (1 / 2) * th.log((1 + x) / (1 - x))


def invert_our_diffeomorphism(sequential):
    # W1 = sequential[0].weight

    def unlinear(y, linear_layer):
        bias = linear_layer.bias
        weight = linear_layer.weight
        return (y - bias) @ weight.inverse().t()

    def my_function(y):
        # x = W1_inv @ inv_tanh(W2_inv @ y)
        x = unlinear(
                inv_tanh(
                    unlinear(y,
                             sequential[-1]
                             )
                    ),
                sequential[0]
                )

        return x
    return my_function


def main():
    dim = 3
    diffeomorphism = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Tanh(),
        nn.Linear(dim, dim)
        )
    nn.init.normal(diffeomorphism[-1].weight)
    nn.init.normal(diffeomorphism[-1].bias)
    weight = diffeomorphism[-1].weight
    bias = diffeomorphism[-1].bias

    inverted = invert_our_diffeomorphism(diffeomorphism)
    input = th.rand(dim, dim)
    # print(input)
    print(diffeomorphism(input))
    print(th.addmm(bias, nn.functional.tanh(input), weight.t()))
    assert isclose(input.numpy(),
                   inverted(diffeomorphism(input)).detach().numpy()).all()


if __name__ == '__main__':
    main()
