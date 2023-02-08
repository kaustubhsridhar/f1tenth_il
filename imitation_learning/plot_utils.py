import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
sns.set_theme(style="darkgrid")

SMALL_SIZE = 12
MEDIUM_SIZE = 17
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_experts(map_name = 'config_example_map', vgain_scales=[0.5, 1.0, 2.0, 3.0], x_lo=None, x_up=None, y_lo=None, y_up=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    trajs = {}
    for vgain_scale in vgain_scales:
        with open(f'logs/{map_name}_expert_waypoint_follower_with_vgain_scale_{vgain_scale}.pkl', 'rb') as f:
            trajs[vgain_scale] = pickle.load(f)
        ax.plot(trajs[vgain_scale]["poses_x"], trajs[vgain_scale]["poses_y"], label=f'{vgain_scale}x top speed')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title(f'Experts with different top speeds in {map_name}')
    ax.legend()
    if x_lo is not None and x_up is not None:
        ax.set_xlim(x_lo, x_up)
    if y_lo is not None and y_up is not None:
        ax.set_ylim(y_lo, y_up)
    plt.show()


