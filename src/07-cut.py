import matplotlib.pyplot as plt
import numpy as np

contour_values = [
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

X_MAX, Y_MAX, Z_MAX = (1036, 1036, 500)
plt.rcParams["figure.figsize"] = (10,10)

angle = 300
elevation = 0

while angle < 0:
    angle += 360

def parabola(p, center_x, center_y, Y):
    X = center_x + (1 / (4 * p)) * ((Y - center_y) ** 2)
    return X

def fill_gap(ax, color, XYZs):
    for n in range(len(XYZs) - 1):
        for i in range(len(XYZs[n][0])):  # 점의 개수
            print("\t" + f"{n}th {i}th point", end="\r")
            XYZ = []
            for axis in range(3):
                element = np.linspace(XYZs[n][axis][i], XYZs[n + 1][axis][i], 50)
                XYZ.append(element)
            X, Y, Z = maskXYZbyAngle(*tuple(XYZ))
            ax.scatter(X, Y, Z, edgecolor=color)
    print("\tDone.")

# new
def tan(angle):
    return np.tan(np.deg2rad(angle))

# new
def line_by_angle(angle):
    def line(X):
        return tan(90+angle) * (X-X_MAX/2)+Y_MAX/2
    return line

# new
def maskXYZbyAngle(X, Y, Z):
    line = line_by_angle(angle)
    if angle <= 180:
        mask = Y < line(X)
    else:
        mask = Y > line(X)

    return X[mask], Y[mask], Z[mask]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_xlim([0, X_MAX])
ax.set_ylim([0, Y_MAX])
ax.set_zlim([0, Z_MAX])

ax.view_init(elev=elevation, azim=angle)

XYZs = []

# draw contours
for v in contour_values:
    Y = np.linspace(0, 1036, 1000)
    X = parabola(v[0], v[1], 1036 / 2, Y)
    Z = np.full((len(X)), v[2])
    XYZs.append((X, Y, Z))

    X, Y, Z = maskXYZbyAngle(X, Y, Z)
    ax.scatter(X, Y, Z, edgecolor="brown", alpha=0.5, zorder=10)

fill_gap(ax, "brown", XYZs)

XYZs = []

for same_altitudes in strike_points:
    XYZ = []
    A, B = same_altitudes
    for axis in range(3):
        element = np.linspace(A[axis], B[axis], 100)
        XYZ.append(element)
    X, Y, Z = tuple(XYZ)
    XYZs.append(tuple(XYZ))

    X, Y, Z = maskXYZbyAngle(X, Y, Z)
    ax.scatter(X, Y, Z, edgecolor="blue", zorder=5)

fill_gap(ax, "blue", XYZs)
plt.savefig(f"../images/cut_0_{angle}.png", bbox_inches="tight")
print("\ndone")
plt.show()
