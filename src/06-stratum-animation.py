import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib import animation

print(sys.argv)
frame_from = int(sys.argv[1])
frame_to = int(sys.argv[2])
print(frame_from, frame_to)

if not frame_to or frame_from >= frame_to:
    raise ValueError("값 제대로 주시와요")

plt.rcParams["figure.figsize"] = (10,10)

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
            X, Y, Z = tuple(XYZ)
            ax.scatter(X, Y, Z, edgecolor=color)
    print("\tDone.")

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

def draw():
    XYZs = []

    # draw contours
    for v in values:
        Y = np.linspace(0, 1036, 1000)
        X = parabola(v[0], v[1], 1036 / 2, Y)
        Z = np.full((len(X)), v[2])
        ax.scatter(X, Y, Z, edgecolor="brown", alpha=0.5, zorder=10)
        XYZs.append((X, Y, Z))

    fill_gap(ax, "brown", XYZs)

    XYZs = []

    for same_altitudes in strike_points:
        XYZ = []
        A, B = same_altitudes
        for axis in range(3):
            element = np.linspace(A[axis], B[axis], 100)
            XYZ.append(element)
        X, Y, Z = tuple(XYZ)
        ax.scatter(X, Y, Z, edgecolor="blue", zorder=5)
        XYZs.append(tuple(XYZ))

    fill_gap(ax, "blue", XYZs)
    print("DRAW OK~")

    return fig,


def animate(i):
    print(f"ANIMATE [{frame_from+i}/{frame_to}]", end="\r")
    ax.view_init(elev=30., azim=(frame_from+i))
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=draw,
                               frames=(frame_to-frame_from), interval=1000, blit=True)

print("ANIMATE [000/000] [OK]")
print("SAVE... ", end="")
# Save
anim.save(f'../images/3d_g2_w_mountain_from{frame_from}.gif', fps=20)
print("[OK]")
