import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


# Goal: draw contours

def parabola(p, center_x, center_y, Y):
    X = center_x + (1 / (4 * p)) * ((Y - center_y) ** 2)
    return X


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlim([0, 1036])
ax.set_ylim([0, 1036])
ax.set_zlim([0, 500])

# img = plt.imread("g2.png")
# ax.imshow(img, extent=[0, 1036, 0, 1036])

values = [
    # (p, x_start, height)
    (60, -10, 100),
    (40, 290, 200),
    (20, 470, 300),
]


DRAW_i = 1
def draw():
    global DRAW_i
    print(f"DRAWING [{DRAW_i}]")
    DRAW_i += 1
    XYZs = []

    for v in values:
        Y = np.arange(0, 1036, 0.5)
        X = parabola(v[0], v[1], 1036 / 2, Y)
        Z = np.full((len(X)), v[2])
        ax.scatter(X, Y, Z, edgecolor="blue")
        XYZs.append((X, Y, Z))

    for n in range(len(XYZs) - 1):
        for i in range(len(XYZs[n][0])):  # 점의 개수
            print("\t"+f"{n}th contour's {i}th point", end="\r")
            XYZ = []
            for axis in range(3):
                element = np.linspace(XYZs[n][axis][i], XYZs[n + 1][axis][i], 50)
                XYZ.append(element)
            X, Y, Z = tuple(XYZ)
            Y = np.full((len(X)), XYZs[n][1][i])
            ax.scatter(X, Y, Z, edgecolor="blue")

    print("DONE")
    return fig,


def animate(i):
    print(f"ANIMATE [{i}/360]", end="\r")
    ax.view_init(elev=30., azim=i)
    return fig,


# Animate
anim = animation.FuncAnimation(fig, animate, init_func=draw,
                               frames=360, interval=20, blit=True)
# Save
anim.save('mpl3d_scatter.gif', fps=30)
print("DONE!!")