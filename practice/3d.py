import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plane_of_z(x, y):
    z = x + y
    return z


x = np.linspace(0, 100, 101)
y = np.linspace(0, 100, 101)
X, Y = np.meshgrid(x, y)  # 격자 그리드 생성

Z = plane_of_z(X, Y)

fig = plt.figure()
fig.set_size_inches(15, 15)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.set_size_inches(15, 15)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')
plt.show()
