import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


def plot_robot(ax, qr, r_robot, color='blue', alpha=1.):
    ax.add_patch(patches.Circle((qr[0], qr[1]), r_robot, color=color, alpha=alpha, 
                                zorder=None))

def plot_object(ax, qo, dim_object, color='blue', alpha=1., zorder=None):
    # qo: [x, y, theta]
    if color =='blue': 
        edgecolor = 'darkblue'
        zorder= 9
    elif color == 'orange':
        edgecolor = 'darkorange'
        zorder = 8
    elif color == 'lightpink': 
        edgecolor='palevioletred'
        zorder = 10
    R = np.array([[np.cos(qo[2]), -np.sin(qo[2])], [np.sin(qo[2]), np.cos(qo[2])]])
    qo_corner = np.array([-dim_object[0]/2, -dim_object[1]/2])
    qo_corner = qo[:2] + np.dot(R, qo_corner)
    ax.add_patch(patches.Rectangle(qo_corner, dim_object[0], dim_object[1], 
                                   angle=np.rad2deg(qo[2]), facecolor=color, 
                                   edgecolor=edgecolor, linewidth=0.5, zorder=zorder,
                                   alpha=alpha))

def plot_object_belief(ax, bo, w, dim_obj, color='red', alpha_factor=10):
    # b: list of [x, y, theta]
    # num particles with w < 1e-3
    print(np.sum(w < 1e-4))
    for qo, wo in zip(bo, w):
        if wo < 1e-4: 
            alpha = 0.2
            color = 'orange'
        else: 
            color = 'blue'
            alpha_factor = 2. 
            alpha = np.min([1,wo*alpha_factor])
            alpha = np.max([0.1, alpha])
            # alpha=0.3
        plot_object(ax, qo, dim_obj, color=color, alpha=alpha)