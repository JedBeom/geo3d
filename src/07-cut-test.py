import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (10, 10)

def curve(X):
    return (1 / 16) * (X - 250) ** 2

def tan(angle):
    return np.tan(np.deg2rad(angle))

def line_by_angle(angle):
    def line(X):
        return tan(90+angle) * (X-250)+250
    return line

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_xlim([0, 500])
ax.set_ylim([0, 500])
ax.set_zlim([0, 500])

angle = 270

ax.view_init(elev=90., azim=angle)

X = np.linspace(0, 500, 500)
Y = np.linspace(0, 500, 500)

X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)

x_min, x_max = (0, 500)
y_min, y_max = (0, 500)

if angle == 0:
    x_max = 250
    Y = Y[X <= x_max]
    Z = Z[X <= x_max]
    X = X[X <= x_max]
elif angle <= 180:
    line = line_by_angle(angle)
    mask = Y < line(X)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]
else:
    line = line_by_angle(angle)
    mask = Y > line(X)
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]


ax.scatter(X, Y, Z, edgecolor="#fd99e1")

plt.show()
