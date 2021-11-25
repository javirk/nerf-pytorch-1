import matplotlib.pyplot as plt
from itertools import cycle


def plot_trajectory(r_hist, m_hist=None, b_pos=0):
    r_hist = r_hist.detach().cpu().numpy()
    cycol = cycle("bgrcmykw")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(r_hist.shape[1]):
        col = next(cycol)
        ax.scatter(r_hist[b_pos, i, 0], r_hist[b_pos, i, 1], r_hist[b_pos, i, 2], color=col)
        if m_hist is not None:
            ax.quiver(*r_hist[b_pos, i], m_hist[b_pos, i, 0] / 20, m_hist[b_pos, i, 1] / 20, m_hist[b_pos, i, 2] / 20,
                      colors=col)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    return fig, ax