import torch as th
import torch.nn as nn

from numpy import allclose

th.manual_seed(1337)


def inv_tanh(x):
    """The inverse of the tanh function."""
    return (1 / 2) * th.log((1 + x) / (1 - x))


def invert_our_diffeomorphism(sequential):
    """Return inverse function of W2 @ tanh(W1 @ x + b1) + b2"""
    def unlinear(y, linear_layer):
        bias = linear_layer.bias
        weight = linear_layer.weight
        return (y - bias) @ weight.inverse().t()

    def my_function(y):
        # x = unlinear(
        #         inv_tanh(
        #             unlinear(y,
        #                      sequential[-1]
        #                      )
        #             ),
        #         sequential[0]
        #         )
        x = unlinear(y, sequential[0])

        return x
    return my_function


def main():
    dim = 3
    diffeomorphism = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Tanh(),
        nn.Linear(dim, dim)
        )
    nn.init.normal_(diffeomorphism[-1].weight)
    nn.init.normal_(diffeomorphism[-1].bias)
    weight = diffeomorphism[-1].weight
    bias = diffeomorphism[-1].bias

    inverted = invert_our_diffeomorphism(diffeomorphism)
    data = th.rand(dim, dim)
    # print(data)
    print(diffeomorphism(data))
    print(th.addmm(bias, nn.functional.tanh(data), weight.t()))
    assert allclose(data.numpy(),
                    inverted(diffeomorphism(data)).detach().numpy())


if __name__ == '__main__':
    main()
