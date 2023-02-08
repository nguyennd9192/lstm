
from general_lib import *
import matplotlib.pyplot as plt
import gc
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def release_mem(fig):
    fig.clf()
    plt.close()
    gc.collect()

def ax_setting():
    plt.style.use('default')
    plt.tick_params(axis='x', which='major', labelsize=15)
    plt.tick_params(axis='y', which='major', labelsize=15)

def get_normcolor(c_value, v_range=None):
    if v_range is None:
        vmin = float(min(c_value))
        vmax = float(max(c_value))
        # cmap = cm.Oranges # jet

    else:
        vmin, vmax = v_range[0], v_range[1]
        
    cmap = cm.RdYlGn # seismic, bwr

    normc = Normalize(vmin=vmin, vmax=vmax)
    colors = []
    n_point = len(c_value)
    for i in range(n_point):
        value = normc(float(c_value[i]))
        colors.append(cmap(value))
    mappable = cm.ScalarMappable(norm=normc, cmap=cmap)
    return colors, mappable


def scatter_plot(x, y, ax, xvline=None, yhline=None, 
    sigma=None, mode='scatter', lbl=None, name=None, 
    x_label='x', y_label='y', 
    save_file=None, interpolate=False, coloraray=None, mappable=None, 
    xtick_pos=None, xtick_name=None, 

    linestyle='-.', marker='o', title=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9), linewidth=1.0) # 

    if 'scatter' in mode:
        for i in range(len(x)):
            ax.scatter(x[i], y[i], s=50, alpha=0.8, 
                marker=marker, 
                c=coloraray[i], edgecolor="black") 

    if 'line' in mode:
        ax.plot(x, y, alpha=0.8) 

    divider = make_axes_locatable(ax)

    if ax is None:
        cax = divider.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(mappable, cax=cax, shrink=0.6)


    if xvline is not None:
        ax.axvline(x=xvline, linestyle='-.', color='black')

    if yhline is not None:
        ax.axhline(y=yhline, linestyle='-.', color='black')

    if name is not None:
        for i in range(len(x)):
            # only for lattice_constant problem, 1_Ag-H, 10_Ag-He
            # if tmp_check_name(name=name[i]):
               # reduce_name = str(name[i]).split('_')[1]
               # plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)
            if name[i] is not None:
                ax.annotate(name[i], xy=(x[i], y[i]), size=4, c="black")
        
    if xtick_pos is not None:
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_name, rotation=40, size=4)


    plt.title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax_setting()

    # plt.xlim([3.0, 4.5])
    if save_file is not None:
        plt.legend(prop={'size': 16})
        makedirs(save_file)
        plt.savefig(save_file)
        if ax is None:
            release_mem(fig=fig)

    return ax