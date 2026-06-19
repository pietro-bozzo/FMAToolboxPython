''' Plotting utilities for publication grade figures '''

import numpy as np
import numpy.typing as npt
import matplotlib.axes as mpla
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import matplotlib.typing as mplt
import scipy.stats as spst
import scipy as sp
from collections.abc import Iterable
from prompt_toolkit.contrib.regular_languages import validation
from typing import Literal


def adjustAxes(axs:Iterable[mpla.Axes], format:Literal['paper','poster']='paper'):
    # adjust axes properties to emprove figure appearance
    #
    # arguments:
    #     axs        iterable of matplotlib.axes.Axes
    #     format     {'paper','poster'}, controls font sizes and lines' width

    if isinstance(axs,mpla._axes.Axes):
        axs = [axs]
    elif isinstance(axs,np.ndarray):
        axs = axs.ravel()

    lw = 1 if format == 'paper' else 2
    axw = 1.3 if format == 'paper' else 2.1
    axtick = 9 if format == 'paper' else 14
    axfont = 10 if format == 'paper' else 18

    for ax in axs:
        
        # remove upper and right borders
        ax.spines[['right','top']].set_visible(False)

        # adjust thickness and font size
        ax.spines[['bottom','left']].set_linewidth(lw)
        ax.tick_params(width=axw,labelsize=axtick)
        ax.xaxis.label.set_fontsize(axfont)
        ax.yaxis.label.set_fontsize(axfont)

    return


def makeFigure(title:str=None, n:tuple[int,int]|list[int]=[1,1], size:tuple[float,float]=[20,10], format:Literal['paper','poster']='paper'):
    # make a figure
    #
    # arguments:
    #     title      string, figure title
    #     n          (2,1) int = [1,1], number of subplots rows and columns
    #     size       (2,1) float = [20,10], figure size (cm)
    #     format     {'paper','poster'}, increases figure size, font sizes, and axes lines' width
    #
    # output:
    #     fig        matplotlib figure
    #     axs        iterable of matplotlib.axes.Axes

    cm = 1 / 2.54 # inches to centimeter conversion factor
    if format == 'poster':
        size = [s*2.5 for s in size]
    fig, axs = plt.subplots(n[0],n[1],figsize=[size[0]*cm,size[1]*cm],constrained_layout=True)

    # promote single axis to sequence
    if isinstance(axs,mpla._axes.Axes):
        axs = [axs]

    fig.suptitle(title)
    adjustAxes(axs,format)

    return fig, axs


def setProp(axs:Iterable[mpla.Axes], xlabelcolor:dict[int,mplt.ColorType]=None, xtickvisible:dict[int,bool]=None, **kwargs):
    # set multiple axes properties at once
    #
    # arguments:
    #     axs             iterable of matplotlib.axes.Axes
    #     xlabelcolor     dict of {index: color}, color for the i-th xlabel, where index is i
    #     xtickvisible    dict of {index: bool}, whether to show the i-th major xtick, where index is i
    #     **kwargs        all extra key-word arguments are passed to matplotlib.pyplot.set()

    # promote single axis to sequence
    if isinstance(axs,mpla._axes.Axes):
        axs = [axs]
    elif isinstance(axs,np.ndarray):
        axs = axs.ravel()

    for ax in axs:

        if xlabelcolor is not None:
            for i, label in enumerate(ax.get_xticklabels()):
                if i in xlabelcolor:
                    label.set_color(xlabelcolor[i])

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
        fig.savefig(str(fname)+'.'+f,transparent=True,bbox_inches='tight',pad_inches=0,format=f,dpi=200)

    return


def plotXY(data,start=None,stop=None,color=None,label=None,ax=None):
    # plot columns of 'data', interpreting the first as the x axis and all others as y values

    data = np.array(data, ndmin=2)
    x = data[:,0]
    if ax is None:
        ax = plt.gca()

    valid = np.full(x.shape,True)
    if start is not None:
        valid[x < start] = False
    if stop is not None:
        valid[x > stop] = False

    n_lines = data.shape[1] - 1
    if color is None or isinstance(color,str):
        color = [color] * n_lines
    if label is None or isinstance(label,str):
        label = [label] * n_lines

    for i in range(n_lines):
        ax.plot(x[valid], data[valid,i+1], color=color[i], label=label[i])

    return


def plotColorMap(data: npt.NDArray[np.floating], vmin: float = None, vmax: float = None, zscore = None, sortby = None, sortax: int = 0,
                 xzoom: float = None, yzoom: float = None, x = None, y = None, aspect: float = 3/4, ax = None):
    # plot a 2D array as a colormap with optional normalization, sorting, and resampling
    #
    # arguments:
    #     data            (n,m) float, data to visualize, rows correspond to first dimension
    #     vmin, vmax      float = None, lower / upper bound of the colormap, if None uses autoscale
    #     zscore          int | 'all' = None, if int, specifies axis along which to z-score `data`; if 'all', compute z-score over whole array,
    #                     if None, no normalization is applied
    #     sortby          callable | 'peak' = None, optional method used to sort `data` along `sortax` (after optional z-scoring), either:
    #                     - 'peak', sort rows or columns by the index of their maximum value along the opposite axis.
    #                     - callable, must have signature ``f(data) -> array_like`` and return a 1D array used to sort data along `sortax`
    #     sortax          int = 0, axis along which sorting is performed
    #     xzoom, yzoom    float = None, optional horizontal / vertical resampling factor passed to ``scipy.ndimage.zoom`` (after sorting)
    #     x, y            (:,) float, coordinates corresponding to columns and rows of `data`, defaults are range(m) and range(n)
    #     aspect          float = 3/4, image aspect ratio
    #     ax              matplotlib.axes.Axes = matplotlib.pyplot.gca(), axes to plot in

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
        x = [0,n_x-1]
        dx = 0.5
    else:
        dx = (x[-1] - x[0]) / (data.shape[1] - 1) / 2 # here use post-zoom shape
    if y is None:
        y = [0,n_y-1]
        dy = 0.5
    else:
        dy = (y[-1] - y[0]) / (data.shape[0] - 1) / 2
    if ax is None:
        ax = plt.gca()

    ax.set_aspect(aspect)
    im = ax.imshow(data,aspect='auto',vmin=vmin,vmax=vmax,origin='lower',extent=[x[0]-dx,x[-1]+dx,y[0]-dy,y[-1]+dy])

    return im


def semPlot(x, y, ci = None, alpha = 0.5, zscore: bool = False, color = None, label: str = None, lprop: dict = None, aprop: dict = None, ax: mpla.Axes = None):
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
    #     ax        matplotlib.axes.Axes = matplotlib.pyplot.gca(), axes to plot in
    
    y = np.array(y)
    y = y[~np.isnan(y).all(axis=1)] # ŕemove full-nan rows

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
    y_line = np.nanmean(y,axis=0)
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


def boxPlot(data, x=None, color=None, label=None, ax:mpla.Axes=None):

    if ax is None:
        ax = plt.gca()

    # remove nans
    if isinstance(data,np.ndarray):
        if data.ndim == 1:
            data = data[~np.isnan(data)]
        else:
            data = [column[~np.isnan(column)] for column in data.T]
    else:
        data = [np.array(d)[~np.isnan(d)] for d in data]

    try:
        color = mplc.to_rgba(color)
        color = [color] * len(data)
    except:
        pass

    lw = ax.spines["left"].get_linewidth() * 0.8
    mksz = ax.spines["left"].get_linewidth() * 2
    medianprops = {'linewidth': lw}
    boxprops = {'linewidth': lw}
    flierprops={'marker':'.', 'markerfacecolor': 'black', 'markersize': mksz}

    bp = ax.boxplot(data,patch_artist=True,positions=x,boxprops=boxprops,medianprops=medianprops,whiskerprops={'linewidth':lw},capprops={'linewidth':lw},flierprops=flierprops)
    if color is not None:
        for box, col in zip(bp["boxes"],color):
            r, g, b, a = mplc.to_rgba(col)
            boxprops['facecolor'] = (r, g, b, a * 0.2)
            box.set(facecolor=(r, g, b, a*0.2),edgecolor=col)
        for median, col in zip(bp['medians'],color):
            median.set_color(col)

    if label is not None:
        if x is None:
            x = np.arange(1,len(label)+1)
        ax.set_xticks(x,label)

    return


def pBar(p, x = None, alpha=0.05, dy=1, draw=(False,True,True,True), ax:mpla.Axes=None):
    # draw horizontal bars indicating significant differences between distributions
    #
    # arguments:
    #     p        (n,3) float, each row is [i,j,pij], where pij is the p value for a test comparing i-th and j-th populations
    #     x        (n,) float = range(n), x coordinates for populations
    #     alpha    float = 0.05, false-discovery tolerance level
    #     dy       float = 1, scale vertical distances between bars
    #     draw     (4,) bool = [False,True,True,True], draw flags for [n.s., *, **, ***]
    #     ax       matplotlib.axes.Axes = matplotlib.pyplot.gca(), axes to plot in

    p = np.array(p,ndmin=2)
    x = np.arange(p.shape[0]) if x is None else np.asarray(x)
    if ax is None:
        ax = plt.gca()
    
    dx = np.diff(ax.get_xlim())[0] / 500
    y_lim = ax.get_ylim()
    height = y_lim[1]
    dy = np.diff(y_lim)[0] / 30 * dy

    # sort according to distance: nearby pairs first, then second neighbours and so on
    distances = np.round(np.diff(x[p[:,0:2].astype(int)],axis=1).ravel(),10)
    order = np.lexsort((x[p[:,0]],distances))
    p = p[order]

    # significance level
    h = p[:,2].copy()
    if alpha != -1:
        h[p[:,2] < alpha] = 1
        h[p[:,2] < alpha/5] = 2
        h[p[:,2] < alpha/50] = 3
        h[p[:,2] >= alpha] = 0

    lw = ax.spines["left"].get_linewidth()
    fontsz = ax.xaxis.label.get_fontsize() * .8
    def _plot_line(ax, x, y, dy, p, last_p, t):
        if last_p > p[0]: # increase height not to overlap lines
            y = y + dy*3.5
        ax.plot([x[0],x[0],x[1],x[1]],[y-dy,y,y,y-dy],color='k',lw=lw)
        ax.text(np.mean(x),y+0.8*dy,t,ha='center',va='center',color='k',size=fontsz)
        last_p = p[1]
        return y, last_p

    last_i = -np.inf
    for i in range(len(p)):
        x_coord = [x[int(p[i,0])]+dx, x[int(p[i,1])]-dx]
        if h[i] == 3 and draw[3]:
            height, last_i = _plot_line(ax,x_coord,height,dy,p[i,0:2],last_i,'***')
        elif h[i] >= 2 and draw[2]:
            height, last_i = _plot_line(ax,x_coord,height,dy,p[i,0:2],last_i,'**')
        elif h[i] >= 1 and draw[1]:
            height, last_i = _plot_line(ax,x_coord,height,dy,p[i,0:2],last_i,'*')
        elif h[i] == 0 and draw[0]:
            height, last_i = _plot_line(ax,x_coord,height,dy,p[i,0:2],last_i,'n. s.')

    ax.set_ylim(y_lim[0],height+dy*5)

    return


def pHorzLine(p,t=None,dy=None,color=None,ax=None):
    # p: (n_times, n_cond)

    p = np.asarray(p).astype(float)
    if p.ndim == 1:
        p = p.reshape(-1,1)
    if t is None:
        t = range(p.shape[0])
    else:
        t = np.array(t)
    if ax is None:
        ax = plt.gca()
    y_lim = ax.get_ylim()
    y = y_lim[1]
    if dy is None:
        dy = np.diff(y_lim)[0] / 20
    if color is None or isinstance(color,str):
        color = [color] * p.shape[1]

    dt = (t[1] - t[0]) / 2
    t = np.stack((t-dt,t+dt)).ravel('F')
    for i, this_p in enumerate(p.T):
        if this_p.any():
            this_p[this_p==0] = np.nan
            this_p = np.stack((this_p,this_p))
            ax.plot(t,this_p.ravel('F')*y,color=color[i])
            y = y + dy

    return


def plotIntervals(intervals,alpha=0.3,color='gray',ax=None):

    intervals = np.array(intervals,ndmin=2)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("'intervals' must have shape (n,2)")

    if ax is None:
        ax = plt.gca()

    for start, stop in intervals:
        ax.axvspan(start,stop,alpha=alpha,color=color)


def plotPDF(x, log:bool=False, bandwidth:float|str=None, eps:float=1e-12, n_points:int=50, color=None, label=None, ax=None, **plot_kwargs):

    if bandwidth is None:
        bandwidth = 'scott'
    if ax is None:
        ax = plt.gca()
    if isinstance(x,tuple):
        if color is None:
            color = [None] * len(x)
        if label is None:
            label = [None] * len(x)
    else:
        x = (x,)
        color = [color]
        label = [label]

    grid = []
    density = []
    for i, data in enumerate(x):
        # cast to array and validate
        data = np.asarray(data)
        if data.size < 2:
            grid.append([])
            density.append([])
            continue
        data = data[~np.isnan(data)] # always ravels input: loosing a capability of gaussian_kde?
        #if data.ndim == 2 and data.shape[1] == 1:
        #    data = data.ravel()

        if log:
            if data.min() <= 0:
                raise ValueError('log-transforming data requires positive values')
            data = np.log(data + eps) # log-transform data
            this_grid = np.linspace(data.min(),data.max(),n_points) # linear grid in log-space
            jacobian = np.exp(this_grid) # jacobian term to transform density back to linear
        else:
            this_grid = np.linspace(data.min(),data.max(),n_points) # linear grid

        kde = sp.stats.gaussian_kde(data,bw_method=bandwidth)
        if log:
            this_density = kde(this_grid) / jacobian
            ax.loglog(jacobian,this_density,color=color[i],label=label[i],**plot_kwargs)
        else:
            this_density = kde(this_grid)
            ax.plot(this_grid,this_density,color=color[i],label=label[i],**plot_kwargs)

    ax.set_yticks([])

    return grid, density