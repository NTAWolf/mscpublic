import os

import numpy as np
import pandas as pd
import seaborn as sns


def xlim_expand(plot, amount=10, relative=False):
    """plot is a handle for a matplotlib plot
    Increases the xlim by amount in both sides
    If relative is True, multiply the y range by amount
    and expand lims as needed.
    """
    lo, hi = plot.get_xlim()
    if relative:
        diff = hi - lo
        newdiff = amount * diff
        amount = (newdiff - diff)/2

    plot.set_xlim((lo-amount, hi+amount))

def ylim_expand(plot, amount=10, relative=False):
    """plot is a handle for a matplotlib plot
    Increases the ylim by amount in top and bottom
    If relative is True, multiply the y range by amount
    and expand lims as needed.
    """
    lo, hi = plot.get_ylim()
    if relative:
        diff = hi - lo
        newdiff = amount * diff
        amount = (newdiff - diff)/2

    plot.set_ylim((lo-amount, hi+amount))

def lim_expand(plot, x_amount=10, y_amount=None, relative=False):
    """
    plot is a matplotlib.Figure (or something like that)
    x_amount is the amount by which xlims should be increased.
    If y_amount is undefined, x_amount is applied to the y-axis
        as well.
    If relative is True, multiply the ranges by amount
        and expand lims as needed.
    """
    xlim_expand(plot, x_amount, relative)
    ylim_expand(plot, y_amount or x_amount, relative)


# FIGPATH = os.environ['HOME'] + '/Dropbox/DTU/4th Semester/figs/'
PATH_THESIS = os.environ['HOME'] + "/Dropbox/Apps/ShareLaTeX/Master's thesis/figs/"
PATH_WEEKLY = os.environ['HOME'] + "/Dropbox/Apps/ShareLaTeX/Weekly prez/figs/"
PATH_DMRI = os.environ['HOME'] + "/Dropbox/Apps/ShareLaTeX/DMRI 4 prez/figs/"

def save_fig(path, tight=True, target='thesis', **savefig_kwargs):
    """Save tight_layout version of current figure in path.
    If the dirs in path do not exist, ask if they should be created.
    """
    if target == 'thesis':
        base = PATH_THESIS
    elif target.lower().startswith('week'):
        base = PATH_WEEKLY
    elif target.lower().startswith('d'):
        base = PATH_DMRI
    path = os.path.join(base, path)
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        if input("Saving fig {}. Create new directory {}? [y]/n > ".format(
                 os.path.basename(path), path_dir)) == 'n':
            return
        os.makedirs(path_dir)

    if tight:
        sns.plt.tight_layout()
    sns.plt.savefig(path, **savefig_kwargs)


