import matplotlib.pyplot as plt
import numpy as np

def parabola(p, center_x, center_y, Y):
    X = center_x + (1 / (4 * p)) * ((Y - center_y) ** 2)
    return X

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlim([0, 1036])
ax.set_ylim([0, 1036])
ax.set_zlim([0, 500])

img = plt.imread("../images/g2.png")
X, Y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
img = img[-Y,X]
Z = np.zeros_like(X)
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, facecolors=img)
plt.savefig("../images/3d_g2_fixed.png", bbox_inches="tight")
print("\ndone")
