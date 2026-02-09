''' Plotting utilities for publication grade figures ''' 

import matplotlib.pyplot


def adjustAxes(axs):
    # adjust axes properties to emprove figure appearance
    #
    # arguments:
    #     axs    collection of matplotlib axes

    if isinstance(axs,matplotlib.axes._axes.Axes):
        axs = [axs]

    for ax in axs:
        
        # remove upper and right borders
        ax.spines[['right', 'top']].set_visible(False)

        # adjust thickness
        for spine in ['top','bottom','left','right']:
            ax.spines[spine].set_linewidth(1.3)
        ax.tick_params(width=1.3)

    return


def makeFigure(title,n=[1,1],size=[20,10]):
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
    fig, axs = matplotlib.pyplot.subplots(n[0],n[1],figsize=[size[0]*cm,size[1]*cm],constrained_layout=True)

    # promote single axis to iterable
    if isinstance(axs,matplotlib.axes._axes.Axes):
        axs = [axs]

    fig.suptitle(title)
    adjustAxes(axs)

    return fig, axs