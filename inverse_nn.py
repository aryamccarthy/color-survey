import torch as th
import torch.nn as nn

from numpy import allclose
from tqdm import tqdm

th.manual_seed(1337)



def print(*xs):
    tqdm.write(" ".join([str(x) for x in xs]))



def inv_tanh(x):
    """The inverse of the tanh function."""
    return (th.log1p(x) - th.log1p(-x)) / 2
    # return (1 / 2) * th.log((1 + x) / (1 - x))


def invert_our_diffeomorphism(sequential):
    """Return inverse function of W2 @ tanh(W1 @ x + b1) + b2"""
    def unlinear(y, linear_layer):
        bias = linear_layer.bias
        weight = linear_layer.weight
        return (y - bias) @ weight.inverse().t()

    def my_function(y):
        print("y: ", y)
        for i, layer in enumerate(reversed(sequential)):
            if isinstance(layer, th.nn.Linear):
                y = unlinear(y, layer)
            elif isinstance(layer, th.nn.Tanh):
                y = inv_tanh(y)
            else:
                raise ValueError("What kind of diffeomorphism did you make?!")
            assert not th.isnan(y).any()
            print(f"l{i}: ", y)
        x = y
        print("x: ", x)
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
