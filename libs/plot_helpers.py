import matplotlib.pyplot as plt
from itertools import cycle, combinations
import numpy as np


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


def plot_tetra(data, tetra_hist, r_hist, m_hist=None, b_pos=0):
    tetra_hist = tetra_hist.detach().cpu()
    r_hist = r_hist.detach().cpu()
    if m_hist is not None:
        m_hist = m_hist.detach().cpu()

    if len(tetra_hist.shape) == 1:
        tetra_hist = tetra_hist[:, None]
        r_hist = r_hist[:, None]
        if m_hist is not None:
            m_hist = m_hist[:, None]

    cycol = cycle("bgrcmykw")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(r_hist.shape[1]):
        col = next(cycol)
        # col = 'b'
        verts = data.pos[data.tetra[:, tetra_hist[b_pos, i]]].cpu().numpy()
        lines = combinations(verts, 2)
        for x in lines:
            line = np.transpose(np.array(x))
            ax.plot3D(line[0], line[1], line[2], c=col)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.scatter(r_hist[b_pos, i, 0], r_hist[b_pos, i, 1], r_hist[b_pos, i, 2], color=col)
        if m_hist is not None:
            ax.quiver(*r_hist[b_pos, i], m_hist[b_pos, i, 0] / 20, m_hist[b_pos, i, 1] / 20, m_hist[b_pos, i, 2] / 20,
                      colors=col)
        ax.text(verts[0,0], verts[0,1], verts[0,2], tetra_hist[b_pos, i].item(), color=col)
