import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = (10,10)

def parabola(p, center_x, center_y, Y):
    X = center_x + (1 / (4 * p)) * ((Y - center_y) ** 2)
    return X

values = [
    # (p, x_start, height)
    (60, -10, 100),
    (40, 290, 200),
    (20, 470, 300),
]

strike_points = [
    # 100m
    ((157, 287, 100), (157, 749, 100)),
    # 200m
    ((443, 342, 200), (443, 694, 200)),
    # 300m
    ((592, 410, 300), (587, 628, 300)),
]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlim([0, 1036])
ax.set_ylim([0, 1036])
ax.set_zlim([0, 500])

'''
# draw an image
img = plt.imread("../images/g2.png")
X, Y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
img = img[-Y, X]
Z = np.zeros_like(X)
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, facecolors=img, zorder=1)
'''

XYZs = []

# draw contours
for v in values:
    Y = np.linspace(0, 1036, 1000)
    X = parabola(v[0], v[1], 1036 / 2, Y)
    Z = np.full((len(X)), v[2])
    ax.scatter(X, Y, Z, edgecolor="brown", alpha=0.5, zorder=10)
    XYZs.append((X, Y, Z))

for n in range(len(XYZs) - 1):
    for i in range(len(XYZs[n][0])):  # 점의 개수
        print("\t"+f"{n}th contour's {i}th point", end="\r")
        XYZ = []
        for axis in range(3):
            element = np.linspace(XYZs[n][axis][i], XYZs[n + 1][axis][i], 100)
            XYZ.append(element)
        X, Y, Z = tuple(XYZ)
        Y = np.full((len(X)), XYZs[n][1][i])
        ax.scatter(X, Y, Z, edgecolor="brown", zorder=10)

for same_altitudes in strike_points:
    XYZ = []
    A, B = same_altitudes
    for axis in range(3):
        element = np.linspace(A[axis], B[axis], 100)
        XYZ.append(element)
    X, Y, Z = tuple(XYZ)
    ax.scatter(X, Y, Z, edgecolor="blue", zorder=5)
    XYZs.append(tuple(XYZ))

ax.view_init(elev=-90., azim=0)
plt.savefig("../images/3d_g2_w_strikes_3.png", bbox_inches="tight")
print("\ndone")
plt.show()
