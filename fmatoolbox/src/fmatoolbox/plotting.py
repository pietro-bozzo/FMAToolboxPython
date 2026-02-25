''' Plotting utilities for publication grade figures '''

import numpy as np
import numpy.typing as npt
import matplotlib.axes as matpla
import matplotlib.pyplot as plt
import scipy.stats as spst
import scipy as sp


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
        ax.spines[['bottom','left']].set_linewidth(1)
        ax.tick_params(width=1.3,labelsize=8)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)

    return


def makeFigure(title: str, n: list[int] = [1,1], size: list[float] = [20,10]):
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


def saveFigure(fig,fname,format):

    # promote single format to iterable
    if isinstance(format,str):
        format = [format]
    for f in format:
        fig.savefig(fname+'.'+f,transparent=True,bbox_inches='tight',pad_inches=0,format=f,dpi=200)

    return


def plotColorMap(data: npt.NDArray[np.floating], vmin: float = None, vmax: float = None, zscore = None, sortby = None, sortax: int = 0,
                 xzoom: float = None, yzoom: float = None, x = None, y = None, aspect: float = 3/4, ax = None):
    """
    Plot a 2D array as a colormap with optional normalization, sorting, and resampling

    Parameters
    ----------
    data : ndarray, shape (M, N)
        two-dimensional array to visualize, rows correspond to first dimension
    vmin, vmax : float, optional
        lower / upper bound of the colormap, if None uses autoscale
    zscore : {int, "all"}, optional
        if integer, specifies axis along which to z-score `data`; if "all", compute z-scores over whole array;
        if None, no normalization is applied
    sortby : {"peak", callable}, optional
        method used to sort `data` along `sortax` (after optional z-scoring), either:
        - "peak", sort rows or columns by the index of their maximum value along the opposite axis.
        - callable, must have signature ``f(data) -> array_like`` and return a 1D array used to sort data along `sortax`
        - None, no sorting is performed
    sortax : {0, 1}
        axis along which sorting is applied
    xzoom, yzoom : float, optional
        horizontal / vertical resampling factor passed to ``scipy.ndimage.zoom`` (after sorting); if None,
        no resampling is performed
    x, y : ndarray, shape (L,), optional
        coordinates corresponding to columns rows of `data`
    aspect : float = 3/4
        image aspect ratio
    ax : matplotlib.axes.Axes, optional
        Axes object in which to draw the plot; if None, uses ``matplotlib.pyplot.gca()``.
    """

    # store original shape in case data neds to be zoomed
    n_y, n_x = data.shape

    if zscore is not None:
        if zscore == 'all':
            zscore = None
        data = spst.zscore(data,axis=zscore)

    if sortby is not None:
        if sortby == 'peak':
            sortby = lambda x : np.argsort(np.argmax(x,1-sortax))
        sort_idx = sortby(data)
        data = data[sort_idx,:] if sortax == 0 else data[:,sort_idx]

    if xzoom is not None or yzoom is not None:
        xzoom = 1 if xzoom is None else xzoom
        yzoom = 1 if yzoom is None else yzoom
        data = sp.ndimage.zoom(data,(yzoom,xzoom))

    if x is None:
        x = [0,n_x]
        dx = 0.5
    else:
        dx = (x[-1] - x[0]) / (data.shape[1] - 1) / 2 # here use post-zoom shape
    if y is None:
        y = [0,n_y]
        dy = 0.5
    else:
        dy = (y[-1] - y[0]) / (data.shape[0] - 1) / 2
    if ax is None:
        ax = plt.gca()

    ax.set_aspect(aspect)
    im = ax.imshow(data,aspect='auto',vmin=vmin,vmax=vmax,origin='lower',extent=[x[0]-dx,x[-1]+dx,y[0]-dy,y[-1]+dy])

    return im


def semPlot(x, y, ci = None, alpha = 0.5, zscore: bool = False, color = None, label: str = None, ax: matpla.Axes = None):
    # plot mean +/- s.e.m. of matrix data
    #
    # arguments:
    #     x         (n) float, x coordinates
    #     y         (:,n) float, data to plot, each column corresponds to a value of x
    #     ci        function, used to compute confidence intervals for every column of y, must have signature:  low, high = ci(y)
    #     alpha     float = 0.5, shaded area transparency value
    #     zscore    bool = False, if True, z-score w.r.t. average of y
    #     color     color = None
    #     label     str = None, legend label for line
    #     ax        Axes = plt.gca(), axes to plot in

    y = np.array(y)

    if ci is None:
        if y.shape[0] < 500:
            ci = lambda x : spst.bootstrap((x,),np.mean,n_resamples=500,vectorized=True,paired=True).confidence_interval
        else:
            ci = lambda x : (x.mean(axis=0) - x.std(axis=0,ddof=1)/np.sqrt(x.shape[0]), x.mean(axis=0) + x.std(axis=0,ddof=1)/np.sqrt(x.shape[0]))
    if ax is None:
        ax = plt.gca()

    # statistic value for each column
    y_line = y.mean(axis=0)
    # statistic confidence interval for each column
    y_low, y_high = ci(y)

    # z-score
    if zscore:
        # mean and standard deviation of average y
        m = y_line.mean()
        s = y_line.std(ddof=1)
        # z-score s.e.m.
        dy_low = (y_line - y_low) / s
        dy_high = (y_high - y_line) / s
        # z-score average
        y_line = (y_line - m) / s
        y_low = y_line - dy_low
        y_high = y_line + dy_high

    ax.plot(x,y_line,label=label,color=color)
    ax.fill_between(x,y_low,y_high,color=color,alpha=alpha,lw=0)

    return