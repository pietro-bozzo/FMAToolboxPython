''' Plotting utilities for publication grade figures '''

import matplotlib.axes as matpla
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def adjustAxes(axs: matpla.Axes):
    # adjust axes properties to emprove figure appearance
    #
    # arguments:
    #     axs    collection of matplotlib axes

    if isinstance(axs,matpla._axes.Axes):
        axs = [axs]

    for ax in axs:
        
        # remove upper and right borders
        ax.spines[['right','top']].set_visible(False)

        # adjust thickness and font size
        ax.spines[['bottom','left']].set_linewidth(1.3)
        ax.tick_params(width=1.3,labelsize=11)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)

    return


def makeFigure(title: str, n: list[int] = [1,1], size: list[int] = [20,10]):
    # make a figure
    #
    # arguments:
    #     title    string, figure title
    #     n        (2,1) int = [1,1], subplots number
    #     size     (2,1) float = [20,10], figure size (cm)
    #
    # output:
    #     fig    matplotlib figure
    #     axs    collection of matplotlib axes

    cm = 1 / 2.54 # inches to centimeter conversion factor
    fig, axs = plt.subplots(n[0],n[1],figsize=[size[0]*cm,size[1]*cm],constrained_layout=True)

    # promote single axis to iterable
    if isinstance(axs,matpla._axes.Axes):
        axs = [axs]

    fig.suptitle(title)
    adjustAxes(axs)

    return fig, axs


def plotColorMap(data: npt.NDArray[np.floating], vmin = None, vmax = None, x = None, y = None, aspect = 3/4, ax = None):
    # plot a colormap
    #
    # arguments:
    #     data      (:,:) float, data to plot
    #     vmin      float = None, colormap lower limit
    #     vmax      float = None, colormap upper limit
    #     x         (:) float = None, x values corresponding to columns of data
    #     y         (:) float = None, y values corresponding to columns of data
    #     aspect    float = 3/4, image aspect ratio
    #     ax        Axes = plt.gca(), axes to plot in

    if x is None:
        x = [0,data.shape[1]]
        dx = 0.5
    else:
        dx = (x[-1] - x[0]) / (data.shape[1] - 1) / 2
    if y is None:
        y = [0,data.shape[0]]
        dy = 0.5
    else:
        dy = (y[-1] - y[0]) / (data.shape[0] - 1) / 2
    if ax is None:
        ax = plt.gca()

    ax.set_aspect(aspect)
    im = ax.imshow(data,aspect='auto',vmin=vmin,vmax=vmax,origin='lower',extent=[x[0]-dx,x[-1]+dx,y[0]-dy,y[-1]+dy])

    return im