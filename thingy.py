# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('text', usetex=True)


# Fixing random state for reproducibility
np.random.seed(19680801)


from string import ascii_uppercase
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import read_data

LabTriple = Tuple[float]
RgbTriple = Tuple[float]

def lab2rgb(lab: LabTriple) -> RgbTriple:
    y = (lab[0] + 16) / 116
    x = lab[1] / 500 + y
    z = y - lab[2] / 200

    x = 0.95047 * (x ** 3 if (x**3 > 0.008856) else (x - 16/116) / 7.787)
    y = 1.00000 * (y ** 3 if (y ** 3 > 0.008856) else (y - 16/116) / 7.787)
    z = 1.08883 * ( z ** 3 if (z ** 3 > 0.008856) else (z - 16/116) / 7.787)

    r = x *  3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y *  1.8758 + z *  0.0415
    b = x *  0.0557 + y * -0.2040 + z *  1.0570

    r = (1.055 * r ** (1/2.4) - 0.055) if (r > 0.0031308) else 12.92 * r
    g = (1.055 * g ** (1/2.4) - 0.055) if (g > 0.0031308) else 12.92 * g
    b = (1.055 * b ** (1/2.4) - 0.055) if (b > 0.0031308) else 12.92 * b

    return (
        int(max(0, min(1, r)) * 255),
        int(max(0, min(1, g)) * 255),
        int(max(0, min(1, b)) * 255),
    )

def produce_grid():
    data = read_data.make_color_lookup()
    chips = {}
    for chip_id, row in data.iterrows():
        letter, *number = chip_id  # Split off first character
        chips[letter, int(''.join(number))] = (tuple(row), lab2rgb(row))
    return chips
    # grid = [[(0,0,0) for _ in range(41)] for _ in range(10)]
    # for (letter, number), rgb in chips.items():
    #     letter_num = ascii_uppercase.index(letter)
    #     grid[letter_num][number] = rgb
    # plt.imshow(grid)
    # plt.show()

chips = produce_grid()

    # data = 
    # for chip_id, row in data.iterrows():
    #     # print(row)
    #     print(chip_id, lab2rgb(row))
    # rgb = [lab2rgb(row) for _, row in data.iterrows()]
    # plt.imshow([rgb])
    # plt.show()


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)

ll = []
aa = []
bb = []
rgbs = []
for (lab, rgb) in chips.values():
    l, a, b = lab
    ll.append(l)
    aa.append(a)
    bb.append(b)
    r, g, b = rgb
    rgbs.append([(r / 255), (g / 255), (b / 255)])
ax.plot([0, 0], [0, 0], [0, 100], c='k', linewidth=5)
ax.scatter(aa, bb, ll, c=rgbs, s=70, alpha=0.7)

ax.set_xlabel('$a$')
ax.set_ylabel('$b$')
ax.set_zlabel('$L$')

plt.show()
