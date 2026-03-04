''' Plotting utilities for publication grade figures '''

import numpy as np
import numpy.typing as npt
import matplotlib.axes as matpla
import matplotlib.pyplot as plt
import scipy.stats as spst
import scipy as sp


def adjustAxes(axs: matpla.Axes, format: str = 'paper'):
    # adjust axes properties to emprove figure appearance
    #
    # arguments:
    #     axs        sequence of matplotlib.axes.Axes
    #     format     {'paper','poster'}, controls font sizes and lines' width

    if isinstance(axs,matpla._axes.Axes):
        axs = [axs]

    lw = 1 if format == 'paper' else 2
    axw = 1.3 if format == 'paper' else 2.1
    axtick = 8 if format == 'paper' else 14
    axfont = 9 if format == 'paper' else 18

    for ax in axs:
        
        # remove upper and right borders
        ax.spines[['right','top']].set_visible(False)

        # adjust thickness and font size
        ax.spines[['bottom','left']].set_linewidth(lw)
        ax.tick_params(width=axw,labelsize=axtick)
        ax.xaxis.label.set_fontsize(axfont)
        ax.yaxis.label.set_fontsize(axfont)

    return


def makeFigure(title: str, n: list[int] = [1,1], size: list[float] = [20,10], format: str = 'paper'):
    # make a figure
    #
    # arguments:
    #     title      string, figure title
    #     n          (2,1) int = [1,1], subplots number
    #     size       (2,1) float = [20,10], figure size (cm)
    #     format     {'paper','poster'}, increases figure size,font sizes, and axes lines' width
    #
    # output:
    #     fig        matplotlib figure
    #     axs        sequence of matplotlib.axes.Axes

    cm = 1 / 2.54 # inches to centimeter conversion factor
    if format == 'poster':
        size = [s*2.5 for s in size]
    fig, axs = plt.subplots(n[0],n[1],figsize=[size[0]*cm,size[1]*cm],constrained_layout=True)

    # promote single axis to sequence
    if isinstance(axs,matpla._axes.Axes):
        axs = [axs]

    fig.suptitle(title)
    adjustAxes(axs,format)

    return fig, axs


def set(axs: matpla.Axes,xtickcolors=None,xtickvisible=None,**kwargs):
    # set multiple axes properties at once

    # promote single axis to sequence
    if isinstance(axs,matpla._axes.Axes):
        axs = [axs]

    for ax in axs:

        if xtickcolors is not None:
            for i, label in enumerate(ax.get_xticklabels()):
                if i in xtickcolors:
                    label.set_color(xtickcolors[i])

        if xtickvisible is not None:
            for i, tick in enumerate(ax.xaxis.get_major_ticks()):
                if i in xtickvisible:
                    tick.tick1line.set_visible(xtickvisible[i])

        ax.set(**kwargs)

    return


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


def semPlot(x, y, ci = None, alpha = 0.5, zscore: bool = False, color = None, label: str = None, lprop: dict = None, aprop: dict = None, ax: matpla.Axes = None):
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
    #     lprop     dict = {}, keyword arguments passed to matplotlib.pyplot.plot
    #     aprop     dict = {}, keyword arguments passed to matplotlib.pyplot.fill_between
    #     ax        Axes = plt.gca(), axes to plot in
    
    y = np.array(y)

    # default values
    if lprop is None: lprop = {}
    if aprop is None: aprop = {}
    lprop.setdefault('color',color)
    lprop.setdefault('label',label)
    if not (set(['edgecolor','edgecolors','ec','facecolor','facecolors','fc','color']) & aprop.keys()): aprop['color'] = color
    aprop.setdefault('alpha',alpha)
    aprop.setdefault('lw',0)
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

    ax.plot(x,y_line,**lprop)
    ax.fill_between(x,y_low,y_high,**aprop)

    return