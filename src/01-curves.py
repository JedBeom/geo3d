import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.set_xlim([0, 1036])
ax.set_ylim([0, 1036])

img = plt.imread("g2.png")
ax.imshow(img, extent=[0, 1036, 0, 1036])

values = [
    # (p, x_start, height)
    (60, -10, 100),
    (40, 290, 200),
    (20, 470, 300),
]

def parabola(p, center_x, center_y, Y):
    X = center_x + (1 / (4 * p)) * ((Y - center_y) ** 2)
    return X

for v in values:
    Y = np.arange(0, 1036, 0.5)
    X = parabola(v[0], v[1], 1036 / 2, Y)
    ax.scatter(X, Y, edgecolor="blue")

plt.show()