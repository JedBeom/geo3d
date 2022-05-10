import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


# Goal: draw contours

def parabola(p, center_x, center_y, Y):
    X = center_x + (1 / (4 * p)) * ((Y - center_y) ** 2)
    return X


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlim([0, 1036])
ax.set_ylim([0, 500])
ax.set_zlim([0, 500])
ax.view_init(elev=0., azim=90)

# img = plt.imread("g2.png")
# ax.imshow(img, extent=[0, 1036, 0, 1036])

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


def fill_gap(ax, color, XYZs):
    for n in range(len(XYZs) - 1):
        for i in range(len(XYZs[n][0])):  # 점의 개수
            print("\t" + f"{n}th {i}th point", end="\r")
            XYZ = []
            for axis in range(3):
                element = np.linspace(XYZs[n][axis][i], XYZs[n + 1][axis][i], 50)
                XYZ.append(element)
            X, Y, Z = tuple(XYZ)
            ax.scatter(X, Y, Z, edgecolor=color)
    print("\tDone.")


def draw():
    # contour
    XYZs = []
    for v in contour_values:
        Y = np.arange(0, 1036, 0.5)
        X = parabola(v[0], v[1], 1036 / 2, Y)
        Z = np.full((len(X)), v[2])
        # ax.scatter(X, Y, Z, edgecolor="blue")
        XYZs.append((X, Y, Z))

    fill_gap(ax, "blue", XYZs)

    print()

    # strike
    XYZs = []
    for same_altitudes in strike_points:
        XYZ = []
        A, B = same_altitudes
        for axis in range(3):
            element = np.linspace(A[axis], B[axis], 50)
            XYZ.append(element)
        X, Y, Z = tuple(XYZ)
        # ax.scatter(X, Y, Z, edgecolor="red")
        XYZs.append(tuple(XYZ))

    fill_gap(ax, "red", XYZs)
    return fig,

'''
draw()
plt.show()
'''

angles = np.linspace(-40, 40, 360)

def animate(i):
    print(f"ANIMATE [{i}/360]", end="\r")
    ax.view_init(elev=20, azim=i)
    return fig,


# Animate
anim = animation.FuncAnimation(fig, animate, init_func=draw,
                               frames=360, interval=20, blit=True)
# Save
anim.save('with_strike.gif', fps=30)
print("\nSave Complete.")