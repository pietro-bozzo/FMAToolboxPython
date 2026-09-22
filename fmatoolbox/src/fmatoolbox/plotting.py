"""Plotting utilities for publication grade figures"""

import fmatoolbox
import numpy as np
import matplotlib.axes as mpla
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import scipy as sp
from collections.abc import Iterable, Collection, Sequence
from typing import Literal, Callable
from matplotlib.axes import Axes
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike
from os import PathLike


def adjustAxes(axs:Axes|Iterable[Axes], format:Literal['paper','poster']='paper'):
    # adjust axes properties to improve figure appearance
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
    ax_title_fs = 9 if format == 'paper' else 18
    ax_label_fs = 9 if format == 'paper' else 18
    ax_label_pad = 0.1 if format == 'paper' else 1
    ax_tick_fs = 8 if format == 'paper' else 14
    ax_tick_l = 2 if format == 'paper' else 5
    ax_tick_pad = 1 if format == 'paper' else 2

    for ax in axs:
        
        # remove upper and right borders in catesian axis
        [ax.spines[spine].set_visible(False) for spine in ['right','top'] if spine in ax.spines]

        # adjust thickness and font size
        [ax.spines[spine].set_linewidth(lw) for spine in ['bottom','left','polar'] if spine in ax.spines]
        ax.tick_params(width=axw,labelsize=ax_tick_fs,pad=ax_tick_pad,length=ax_tick_l)
        ax.title.set_fontsize(ax_title_fs) # NOTE: seems not to work
        ax.xaxis.label.set_fontsize(ax_label_fs)
        ax.yaxis.label.set_fontsize(ax_label_fs)
        ax.xaxis.labelpad = ax_label_pad
        ax.yaxis.labelpad = ax_label_pad

    return


def makeFigure(title:str=None, n:tuple[int,int]=[1,1], size:tuple[float,float]=[20,10], projection:str=None, format:Literal['paper','poster']='paper'):
    # make a figure
    #
    # arguments:
    #     title         string, figure title
    #     n             (2,1) int = [1,1], number of subplots rows and columns
    #     size          (2,1) float = [20,10], figure size (cm)
    #     projection    str = None, projection of axes
    #     format        {'paper','poster'}, increases figure size, font sizes, and axes lines' width
    #
    # output:
    #     fig           matplotlib figure
    #     axs           iterable of matplotlib.axes.Axes

    cm = 1 / 2.54 # inches to centimeter conversion factor
    if format == 'poster':
        size = [s*2.5 for s in size]
    fig, axs = plt.subplots(n[0],n[1],figsize=[size[0]*cm,size[1]*cm],constrained_layout=True,subplot_kw=dict(projection=projection))

    # promote single axis to sequence
    if isinstance(axs,mpla._axes.Axes):
        axs = [axs]

    fig.suptitle(title)
    adjustAxes(axs,format)

    return fig, axs


def setProp(axs:Axes|Iterable[Axes], xlabelcolor:dict[int,ColorType]=None, xtickvisible:dict[int,bool]=None, **kwargs):
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


def saveFigure(fig, fname:str|PathLike[str], format:str|Iterable[str]):

    # promote single format to iterable
    if isinstance(format,str):
        format = (format,)
    for f in format:
        fig.savefig(str(fname)+'.'+f,transparent=True,bbox_inches='tight',pad_inches=0,format=f,dpi=200)

    return


def setCLim(im:Collection,vmin:float|Sequence[float]=None,vmax:float|Sequence[float]=None):
    """set limits of multiple colormaps

    arguments:
        im            Collection[images]
        vmin, vmax    float | Sequence[float] = None, values to set as min (max) of the colormaps of 'im', either:
                       - None: all images' min (max) will be set as the overall min (max)
                       - float: one value per image or one for all images
    """

    auto = vmin is None or vmax is None
    if auto:
        min_v = np.inf
        max_v = -np.inf
        for image in im:
            clim = image.get_clim()
            min_v = min(min_v,clim[0])
            max_v = max(max_v,clim[1])
        vmin = min_v if vmin is None else vmin
        vmax = max_v if vmax is None else vmax

    if ~isinstance(vmin,Sequence):
        vmin = (vmin,) * len(im)
    if ~isinstance(vmax,Sequence):
        vmax = (vmax,) * len(im)

    for i, image in enumerate(im):
        clim = list(image.get_clim())
        clim[0] = clim[0] if vmin[i] is None else vmin[i]
        clim[1] = clim[1] if vmax[i] is None else vmax[i]
        image.set_clim(clim)

    return


def plot(x, y=None, *args, start=None, stop=None, polar:bool=None, ax:Axes=None, **kwargs):
    """wrapper around `matplotlib.pyplot.plot` with extra functionalities

    arguments:
        x              (n,) float = range(n), x coordinates (optional)
        y              (n,) | (n,:) float, data to plot, each row corresponds to a value of x
        start, stop    float, if given, only plot rows of `y` corresponding to `x` in the interval [start,stop] TO IMPLEMENT
        polar          bool = False, if True, data is expected to span a full circle, `x[-1] + x[1] - x[0]` and `y[0]` are appended
                       to `x` and `y` to close a circular plot, and eventual smoothing (NOT IMPLEMENTED!) is applied in a circular fashion
        ax             matplotlib.axes.Axes = matplotlib.pyplot.gca(), axes to plot in

    note: all extra arguments are passed to `matplotlib.pyplot.plot`
    """

    if ax is None:
        ax = plt.gca()

    if polar:
        # 'x' is optional
        if y is None:
            y = x
            x = None

        y = np.array(y, ndmin=1)
        if y.ndim > 2:
            if all(s == 1 for s in y.shape[2:]):
                y = y[(...,) + (0,) * (y.ndim - 2)]
            else:
                raise ValueError("'y' must be 1d or 2d")
        if x is None: x = np.arange(y.shape[0])

        x = np.append(x, x[-1]+x[1]-x[0])
        y = np.append(y, y[0:1], axis=0)

    args = (x, y, *args) if y is not None else (x, *args)
    return ax.plot(*args, **kwargs)


def plotXY(data, start=None, stop=None, color:ColorType=None, label=None, ax:Axes=None):
    # plot columns of 'data', interpreting the first as the x axis and all others as y values

    data = np.array(data,ndmin=2)
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


def plotColorMap(data:ArrayLike, vmin:float=None, vmax:float=None, zscore=None, omitnan:int=None, sortby:ArrayLike|Callable|str=None,
                 sortax:int=None, xzoom:float=None, yzoom:float=None, smooth=None, alpha=None, x=None, y=None, aspect:float=None, bar:str=None, ax:Axes=None):
    """plot a 2D array as a colormap with optional normalization, sorting, and resampling

    arguments:
        data            (n,m) float, data to visualize, rows of the resulting image correspond to `data`'s first dimension
        vmin, vmax      float = None, lower / upper bound of the colormap, if None uses autoscale
        zscore          int | 'all' = None, if int, specifies axis along which to z-score `data`; if 'all', compute z-score over whole array,
                        if None, no normalization is applied
        omitnan         int = None, axis along which to drop slices of `data` if they contain only nans
        sortby          (:,) int | callable | 'peak' = None, optional method used to sort `data` along `sortax` (after optional z-scoring), either:
                        - 'peak' | 'peak-show', sort rows or columns by the index of their maximum value along the opposite axis; if '-show', also plot peaks
                        - (:,) int, array of indeces to sort data along `sortax`
                        - callable, must have signature ``f(data) -> array_like`` and return a 1D array used to sort data along `sortax`
        sortax          int = 0, axis along which sorting is performed
        xzoom, yzoom    float = None, optional horizontal / vertical resampling factor passed to ``scipy.ndimage.zoom`` (after sorting)
        smooth          float | (float,float) = None, standard deviation of gaussian kernel passed to ``scipy.ndimage.gaussian_filter`` (after zoom)
        alpha           float | (n,m) float = None, transparency mask for `data` (e.g., to show significance of pixels), values must be in [0,1]
        x, y            (:,) float, coordinates corresponding to columns and rows of `data`, defaults are range(m) and range(n)
        aspect          float = 3/4, image aspect ratio
        bar             str = None, if given, draw colorbar next to `ax` with label specified by `bar`, and return both `im` and `cb` objects
        ax              matplotlib.axes.Axes = matplotlib.pyplot.gca(), axes to plot in

    output:
        im              image
        cb              colorbar, optional
    """

    data = np.array(data,ndmin=2)

    # 1. z-score
    if zscore is not None:
        if zscore == 'all':
            zscore = None
        data = sp.stats.zscore(data,axis=zscore,nan_policy='omit')

    # 2. remove nans
    if omitnan is not None:
        keep = ~np.all(np.isnan(data),axis=omitnan)
        data = data[:,keep] if omitnan == 0 else data[keep,:]
    n_y, n_x = data.shape # store shape in case data needs to be zoomed

    # 3. sort rows / columns
    if sortax is None: sortax = 0
    if sortby is not None:
        if isinstance(sortby,str) and sortby.startswith('peak'):
            peaks = np.argmax(data,1-sortax)
            sort_idx = np.argsort(peaks)
        elif callable(sortby):
            sort_idx = sortby(data)
        else:
            sort_idx = sortby
        data = data[sort_idx,:] if sortax == 0 else data[:,sort_idx]

    # 4. zoom
    if xzoom is not None or yzoom is not None:
        xzoom = 1 if xzoom is None else xzoom
        yzoom = 1 if yzoom is None else yzoom
        data = sp.ndimage.zoom(data,(yzoom,xzoom))

    # 5. smooth
    if smooth is not None:
        data = sp.ndimage.gaussian_filter(data,smooth)

    # set up axes limits
    if x is None:
        x = np.arange(n_x)
        dx = 0.5
    else:
        dx = 0.5 if n_x == 1 else (x[-1] - x[0]) / (data.shape[1] - 1) / 2 # here use post-zoom shape
    if y is None:
        y = np.arange(n_y)
        dy = 0.5
    else:
        dy = 0.5 if n_y == 1 else (y[-1] - y[0]) / (data.shape[0] - 1) / 2
    if ax is None:
        ax = plt.gca()

    if aspect is None: aspect = 3 / 4
    ax.set_aspect(aspect)
    im = ax.imshow(data,aspect='auto',vmin=vmin,vmax=vmax,origin='lower',extent=[x[0]-dx,x[-1]+dx,y[0]-dy,y[-1]+dy])
    if alpha is not None:
        im.set_alpha(alpha)
    if bar is not None:
        cb = plt.colorbar(im,label=bar,ax=ax)

    # plot peaks
    if sortby == 'peak-show':
        if sortax == 0:
            peaks = peaks if x is None else x[peaks]
            ax.plot(np.sort(peaks),y,color='r')
        else:
            peaks = peaks if y is None else y[peaks]
            ax.plot(x,np.sort(peaks),color='r')

    if bar is not None:
        return im, cb
    return im


def semPlot(x, y=None, ci:str|Callable=None, zscore:int=None, polar:int=None, smooth:float=None, color:ColorType=None, mode:Literal['area','error','bar']=None,
            alpha:float=None, label:str=None, lprop:dict=None, aprop:dict=None, ax:Axes=None):
    """plot mean +/- confidence intervals of matrix data

    arguments:
        x         (n,) float = range(n), x coordinates (optional)
        y         (:,n) float, data to plot, each column corresponds to a value of x
        ci        callable | 'nansem', used to compute confidence intervals for every column of `y`, either:
                  - 'nansem', standard error of the mean (SEM) for each column of `y`, ignoring missing values
                  - callable, must have signature ``low, high = ci(y)``
        zscore    int, if 1, z-score w.r.t. average of y, if 2, z-score each row of y independently, default is no normalization
        polar     int, if given, data is expected to span a full circle, and can be either:
                  - 1, `x[-1] + x[1] - x[0]` and `y[:,0]` are appended to `x` and `y` to close a circular plot
                  - higher than 1, `polar-1` copies of `y` are appended, to plot multiple periods
        smooth    float = None, gaussian kernel std for smoothing over time; if `polar` is given, circular smoothing is applied
        color     color = None
        mode      str = 'area' | 'error' | 'bar', plot 'ci' either as a shaded area, line with error bars, or bar plot
        alpha     float = 0.5, area transparency value (only for 'area' and 'bar' mode)
        label     str = None, legend label for line
        lprop     dict = {}, keyword arguments passed to matplotlib.pyplot.plot
        aprop     dict = {}, keyword arguments passed to matplotlib.pyplot.fill_between (only for 'area' mode)
        ax        matplotlib.axes.Axes = matplotlib.pyplot.gca(), axes to plot in
    """
    # NOTE: should maybe change y to match plt.plot: each row of 'y' corresponds to a values of 'x'

    # 'x' is optional
    if y is None:
        y = x
        x = None

    # validate 'y'
    y = np.array(y,ndmin=2)
    if y.ndim > 2:
        if all(s == 1 for s in y.shape[2:]):
            y = y[(...,) + (0,) * (y.ndim - 2)]
        else:
            raise ValueError("'y' must be 2d")
    if x is None: x = np.arange(y.shape[1])
    if len(x) != y.shape[1]:
        raise ValueError("'y' must have one column per element of 'x'")

    # default values
    y = y[~np.isnan(y).all(axis=1)] # ŕemove full-nan rows
    if y.size == 0:
        return
    zscore = 0 if zscore is None else int(zscore)
    polar = 0 if polar is None else int(polar)
    if mode is None: mode = 'area' if y.shape[1] > 1 else 'error'
    if alpha is None: alpha = 0.5
    if lprop is None: lprop = {}
    if aprop is None: aprop = {}
    lprop.setdefault('color',color)
    lprop.setdefault('label',label)
    if not (set(['edgecolor','edgecolors','ec','facecolor','facecolors','fc','color']) & aprop.keys()): aprop['color'] = color
    aprop.setdefault('alpha',alpha)
    aprop.setdefault('lw',0)

    if isinstance(ci,str) and ci == 'nansem':
        ci = lambda x : (np.nanmean(x,axis=0) - np.nanstd(x,axis=0,ddof=1)/np.sqrt(np.sum(~np.isnan(x),axis=0)),
                         np.nanmean(x,axis=0) + np.nanstd(x,axis=0,ddof=1)/np.sqrt(np.sum(~np.isnan(x),axis=0)))
    elif ci is None:
        if y.shape[0] == 1:
            ci = lambda x : (x.flatten(), x.flatten())
        elif y.shape[0] < 500:
            ci = lambda x : sp.stats.bootstrap((x,),np.nanmean,n_resamples=500,vectorized=True,paired=True).confidence_interval
        else:
            ci = lambda x : (x.mean(axis=0) - x.std(axis=0,ddof=1)/np.sqrt(x.shape[0]), x.mean(axis=0) + x.std(axis=0,ddof=1)/np.sqrt(x.shape[0]))
    if ax is None:
        ax = plt.gca()

    if zscore == 2:
        # z-score rows of y
        y = sp.stats.zscore(y,axis=1,nan_policy='omit')

    # statistic value for each column
    y_line = np.nanmean(y,axis=0)
    # statistic confidence interval for each column
    y_low, y_high = ci(y)

    if zscore == 1:
        # z-score results w.r.t. average y
        # mean and standard deviation of average y
        m = np.nanmean(y_line)
        s = np.nanstd(y_line,ddof=1)
        # z-score s.e.m.
        dy_low = (y_line - y_low) / s
        dy_high = (y_high - y_line) / s
        # z-score average
        y_line = (y_line - m) / s
        y_low = y_line - dy_low
        y_high = y_line + dy_high

    if smooth is not None:
        smooth_mode = 'wrap' if polar else 'reflect'
        y_line = sp.ndimage.gaussian_filter1d(y_line, smooth, axis=0, mode=smooth_mode)
        y_low = sp.ndimage.gaussian_filter1d(y_low, smooth, axis=0, mode=smooth_mode)
        y_high = sp.ndimage.gaussian_filter1d(y_high, smooth, axis=0, mode=smooth_mode)

    if polar == 1:
        x = np.append(x, x[-1]+x[1]-x[0])
        y_line = np.append(y_line, y_line[0])
        y_low = np.append(y_low, y_low[0])
        y_high = np.append(y_high, y_high[0])
    elif polar > 1:
        n_x = len(x)
        for i in range(polar-1):
            x = np.concatenate((x, x[-n_x:] + 2*np.pi))
            y_line = np.concatenate((y_line, y_line[:n_x]))
            y_low = np.concatenate((y_low, y_low[:n_x]))
            y_high = np.concatenate((y_high, y_high[:n_x]))

    copy_color = aprop['color'] is None
    match mode:
        case 'area':
            l = ax.plot(x,y_line,**lprop)
            if copy_color:
                aprop['color'] = l[0].get_color()
            ax.fill_between(x,y_low,y_high,**aprop)
        case 'error':
            ax.errorbar(x,y_line,yerr=[y_low,y_high],marker='o',**lprop)
        case 'bar':
            width = x[1] - x[0]
            ax.bar(x,y_line,width=width,yerr=[y_line-y_low,y_high-y_line],**aprop)

    return


def boxPlot(data:ArrayLike|Sequence[ArrayLike], x:ArrayLike=None, mode:Literal['box','violin','scatter','link']|Collection[str]=None, color:ColorType|Collection[ColorType]=None,
            label:str|Sequence[str]=None, ax:Axes=None):
    """draw box plots to represent groups of datasets
    note: calls matplotlib's boxplot, which sets xticks

    Args:
        data:   groups of datasets to plot, same format as ``matplotlib.pyplot.boxplot`` `x` argument, an array or a sequence of array-like vectors, one per dataset
        x:      positions for each drawn distribution, defaults to `range(n_data)`
        mode:   one or more of the following options, which will be drawn together:
                - 'box', box-and-whiskers plots per dataset, default
                - 'violin', violin plots per dataset
                - 'scatter', scatter plots of data points of each dataset
                - 'link, lines between data points of each adjacent dataset
        color:  color for boxes, violins, and scatter plots, one per dataset or a single one for all, defaults to blue
        label:  xtick label below each plotted distribution, defaults to no labels
        ax:     axes to plot in, defaults to ``matplotlib.pyplot.gca()``
    """

    # attempt casting to np array
    if hasattr(data, 'to_numpy'): data = data.to_numpy()
    if hasattr(data, 'values'):
        data_temp = data.values
        if isinstance(data_temp, np.ndarray):
            data = data_temp
    # make a copy of 'data' free from nans, keep original 'data' to plot scatter and lines
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data_clean = data[~np.isnan(data)]
            n_data = 1
        elif data.ndim == 2:
            data_clean = [data[~np.isnan(data[:,col]),col] for col in range(data.shape[1])]
            data = data.T
            n_data = len(data)
        else:
            raise ValueError("'data' must be 1d or 2d")
    else:
        data_clean = [np.array(d)[~np.isnan(d)] for d in data]
        n_data = len(data)

    # defaults
    if mode is None: mode = ('box',)
    if isinstance(mode, str): mode = (mode,)
    if ax is None: ax = plt.gca()
    x = np.arange(n_data) if x is None else np.array(x,ndmin=1)
    if color is None: color = ('#1c8dfc',) * n_data # blue
    else:
        try:
            color = mplc.to_rgba(color)
            color = (color,) * n_data
        except: pass
    if len(color) != n_data:
        raise ValueError("'color' must have one element for each dataset in 'data'")

    out = []
    if 'box' in mode:
        lw = ax.spines["left"].get_linewidth() * 0.8
        mksz = ax.spines["left"].get_linewidth() * 2
        medianprops = {'linewidth': lw}
        boxprops = {'linewidth': lw}
        flierprops = {'marker': '.','markerfacecolor': 'black','markersize': mksz}

        bp = ax.boxplot(data_clean,patch_artist=True,positions=x,boxprops=boxprops,medianprops=medianprops,whiskerprops={'linewidth':lw},
                        capprops={'linewidth':lw},flierprops=flierprops)
        for box, col in zip(bp["boxes"],color):
            r, g, b, a = mplc.to_rgba(col)
            boxprops['facecolor'] = (r, g, b, a * 0.2)
            box.set(facecolor=(r, g, b, a*0.2),edgecolor=col)
        for median, col in zip(bp['medians'],color):
            median.set_color(col)
        out.append(bp)

    if 'violin' in mode:
        facecolor = []
        for col in color:
            r, g, b, a = mplc.to_rgba(col)
            facecolor.append((r,g,b,a*0.5))
        vp = ax.violinplot(data_clean,positions=x,side='high',facecolor=facecolor,linecolor=color,showmedians=True,showextrema=False)
        out.append(vp)

    # following plots use original data and are unaffected by nans
    x_drawn = [np.repeat(x[i],len(data[i])) for i in range(n_data)]
    if 'scatter' in mode:
        jitter = lambda a : a + np.random.normal(0, (x[1]-x[0])/15, size=len(a))
        x_drawn = [jitter(xi) for xi in x_drawn]
        out.append( [ax.scatter(x_drawn[i], data[i], c=color[i], alpha=0.7) for i in range(n_data)] )

    if 'link' in mode:
        out.append( [ax.plot(np.stack([x_drawn[i],x_drawn[i+1]]), np.stack([data[i],data[i+1]]), color='gray', alpha=0.5) for i in range(n_data-1)] )

    if label is not None:
        ax.set_xticks(x,label)

    return out


def pBar(p:ArrayLike, x:ArrayLike=None, alpha:float=0.05, dy:float=1, draw:Sequence[bool]=(False,True,True,True),
         ax:Axes=None) -> None:
    """plot horizontal bars and asterisks indicating significant differences between distributions

    Args:
        p:      (n,3) float array, each row is [i,j,pij], where pij is the p value for a test comparing i-th and j-th populations
        x:      (n,) float array, x coordinates for populations, defaults to range(n)
        alpha:  false-discovery tolerance level, defaults to 0.05
        dy:     scale vertical distances between bars, defaults to 1
        draw:   draw flags for [n.s., *, **, ***], defaults to (False,True,True,True)
        ax:     axes to plot in, defaults to ``matplotlib.pyplot.gca()``
    """

    p = np.array(p,ndmin=2)
    if p.shape[1] != 3:
        raise ValueError("'p' must have 3 columns")
    indices = p[:,0:2].astype(int)
    x = np.arange(p.shape[0]) if x is None else np.array(x,ndmin=1)
    if ax is None:
        ax = plt.gca()
    
    dx = np.diff(ax.get_xlim())[0] / 500
    y_lim = ax.get_ylim()
    dy = np.diff(y_lim)[0] / 30 * dy
    height = y_lim[1] + dy

    # sort according to distance: nearby pairs first, then second neighbours and so on
    distances = np.round(np.diff(x[indices],axis=1).ravel(),10)
    order = np.lexsort((x[indices[:,0]],distances))
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
    def _plot_line(ax, x, y, dy, p, last_p, t, single):
        if last_p > p[0]: # increase height not to overlap lines
            y = y + dy*3.5
        if not single:
            ax.plot([x[0],x[0],x[1],x[1]],[y-dy,y,y,y-dy],color='k',lw=lw)
        ax.text(np.mean(x),y+0.8*dy,t,ha='center',va='center',color='k',size=fontsz)
        last_p = p[1]
        return y, last_p

    last_i = -np.inf
    for i in range(len(p)):
        single = p[i,0] == p[i,1] # if True, plot only stars (single population)
        x_coord = [x[int(p[i,0])]+dx, x[int(p[i,1])]-dx]
        if h[i] == 3 and draw[3]:
            height, last_i = _plot_line(ax,x_coord,height,dy,p[i,0:2],last_i,'***',single)
        elif h[i] >= 2 and draw[2]:
            height, last_i = _plot_line(ax,x_coord,height,dy,p[i,0:2],last_i,'**',single)
        elif h[i] >= 1 and draw[1]:
            height, last_i = _plot_line(ax,x_coord,height,dy,p[i,0:2],last_i,'*',single)
        elif h[i] == 0 and draw[0]:
            height, last_i = _plot_line(ax,x_coord,height,dy,p[i,0:2],last_i,'n. s.',single)

    ax.set_ylim(y_lim[0],height+dy*5)

    return


def pHorzLine(p, t=None, dy=None, color:ColorType=None, ax:Axes=None, **kwargs):
    """draw a horizontal line indicating time points where time series passed a statistical test

    arguments:
        p           (t,c) bool, decision of a statistical test, t: number of time points, c: number of conditions
        t           (t,) float = range(n), time points
        dy          float, scale vertical distances between bars
        color       color, line color
        ax          matplotlib.axes.Axes = matplotlib.pyplot.gca(), axes to plot in
        **kwargs    all extra key-word arguments are passed to ``matplotlib.pyplot.plot``
    """

    p = np.asarray(p).astype(float)
    if p.ndim == 1:
        p = p.reshape(-1,1)
    if t is None:
        t = np.arange(p.shape[0])
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
            ax.plot(t, this_p.ravel('F')*y, color=color[i], **kwargs)
            y = y + dy

    return


def plotIntervals(intervals, color:ColorType='gray', alpha=0.3, label:str=None, ax:Axes=None, **plot_kwargs):

    intervals = fmatoolbox.general.consolidateIntervals(intervals)
    if intervals.size == 0:
        return
    if ax is None:
        ax = plt.gca()

    ax.axvspan(intervals[0,0],intervals[0,1],color=color,alpha=alpha,label=label,**plot_kwargs)
    for start, stop in intervals[1:]:
        ax.axvspan(start,stop,color=color,alpha=alpha,**plot_kwargs)


def plotPDF(x, mode:Literal['normal','log','polar']=None, method:Literal['kde','discrete']=None, bandwidth:float|str=None, eps:float=None, n_points:int=None, bins=None,
            norm:Literal['density','max','cdf']=None, color:ColorType=None, alpha:float=None, label=None, ax:Axes=None, **plot_kwargs):
    """estimate and plot probability density function (PDF) of data

    arguments:
        x            (n,) tuple | array, values drawn from n stochastic variables X_i, used to estimate their PDFs
        mode         str = {'normal','log','polar'}, DESCRIBE
        method       str = {'kde','discrete'}, DESCRIBE (only for 'normal' mode)
        bandwidth    float | str = 'scott', bandwidth for gaussian kernel
        eps          float = 1e-12, small value used to avoid log(0)
        n_points     int = 50, number of points used to evaluate PDF
        bins         (:,) float = None, bin edges, if None, bins are linearly spaced between min and max of x
        norm         str = {'density','max'}, normalization mode, 'density' computes PDF, 'max' normalizes its maximum to 1,
                     'cdf' computes cumulative density function (CDF)
        color        color = None, line color
        alpha        float, transparency, default is 0.7 for method 'discrete', 1.0 otherwise
        label        str = None, legend label for line
        ax           matplotlib.axes.Axes = matplotlib.pyplot.gca(), axes to plot in

    output:
        grid         (n,) list of (n_points,) float, values of X_i for which the PDF was evalueated
        density      (n,) list of (n_points,) float, estimated PDFs
    """

    if mode is None: mode = 'normal'
    if method is None: method = 'kde'
    if ax is None: ax = plt.gca()
    if isinstance(x,tuple):
        if color is None or isinstance(color,str):
            color = [color] * len(x)
        if alpha is None or isinstance(alpha,int) or isinstance(alpha,float):
            alpha = [alpha] * len(x)
        if label is None or isinstance(label,str):
            label = [label] * len(x)
    else:
        x = (x,)
        color = [color]
        alpha = [alpha]
        label = [label]

    grid = []
    density = []
    for i, data in enumerate(x):
        g, d = fmatoolbox.analysis.PDF(data,mode=mode,method=method,bandwidth=bandwidth,eps=eps,n_points=n_points,bins=bins,norm=norm)
        grid.append(g)
        density.append(d)
        if len(g) == 0:
            continue
        match mode:
            # 1. real-valued data using gaussian kernel density estimator
            case 'normal':
                if method == 'kde':
                    ax.plot(g,d,color=color[i],label=label[i],**plot_kwargs)
                else:
                    if alpha[i] is None: alpha[i] = 0.7
                    ax.bar(g,d,width=g[1]-g[0],color=color[i],label=label[i],alpha=alpha[i],**plot_kwargs)
            # 2. log-transformed data using gaussian kernel density estimator
            case 'log':
                jacobian = np.exp(g) # jacobian term to transform density back to linear
                ax.loglog(jacobian,d,color=color[i],label=label[i],**plot_kwargs)
            # 3. circular data using von-Mises kernel density estimator
            case 'polar':
                ax.plot(g,d,color=color[i],label=label[i],**plot_kwargs)
    ax.set_yticks([])

    return grid, density


def plotRaster(spikes, ids=None, compact:bool=None, offset:float=None, height:float=None, ax:Axes=None, **plot_kwargs):

    if height is None: height = 1

    spikes = np.asarray(spikes)
    if spikes.ndim == 1:
        times = spikes
        units = np.zeros(len(times))
    else:
        times = spikes[:,0]
        units = spikes[:,1]

    if ids is not None:
        # keep requested neurons
        valid = np.isin(units,ids)
        times = times[valid]
        units = units[valid]

    if compact:
        # relabel units from 1 to N
        _, units = np.unique(units,return_inverse=True)
    if offset:
        units += offset

    if ax is None:
        ax = plt.gca()

    half_height = height / 2
    ax.vlines(times, units-half_height, units+half_height, **plot_kwargs)

    return