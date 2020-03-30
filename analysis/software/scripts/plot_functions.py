from aict_tools.io import read_telescope_data, append_column_to_hdf5
import pandas as pd
from astropy.stats import binom_conf_interval
from astropy.coordinates.angle_utilities import angular_separation
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import PowerNorm
from itertools import cycle
from astropy import units as u
import numpy as np
import argparse
from aict_tools.configuration import AICTConfig
from aict_tools.io import read_data

import matplotlib
#matplotlib.use('pdf')


color_pallete = [
    '#cc2a36',
    '#4f372d',
    '#00a0b0',
    '#edc951',
    '#4ab174',
    '#eb6841',
]
default_cmap = 'RdPu'
main_color = '#4386dd'
main_color_complement = '#d63434'
dark_main_color = '#707070'
color_cycle = cycle(color_pallete)
figsize = (10,6)

def add_colorbar_to_figure(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


def make_default_cta_binning(e_min=0.02 * u.TeV, e_max=200 * u.TeV, centering='log', overflow=False, bins_per_decade=5):
    bin_edges = np.logspace(np.log10(0.002), np.log10(2000), 6 * bins_per_decade + 1)
    idx = np.searchsorted(bin_edges, [e_min.to_value(u.TeV), e_max.to_value(u.TeV)])
    max_idx = min(idx[1] + 1, len(bin_edges) - 1)
    bin_edges = bin_edges[idx[0]:max_idx]
    if overflow:
        bin_edges = np.append(bin_edges, 10000)
        bin_edges = np.append(0, bin_edges)

    if centering == 'log':
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    return bin_edges * u.TeV, bin_centers * u.TeV, bin_widths * u.TeV


def make_linear_binning(x_min, x_max):
    bin_edges = np.linspace(x_min, x_max, x_max-x_min+1)

    idx = np.searchsorted(bin_edges, [x_min, x_max])
    max_idx = min(idx[1] + 1, len(bin_edges) - 1)
    bin_edges = bin_edges[idx[0]:max_idx]

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_centers, bin_widths


def plot_angular_resolution(x, y, name, x2=None, y2=None, log=False, out_file=None, hexbin=True, second_color=None):
    e_min, e_max = 0.005 * u.TeV, 200 * u.TeV
    if log:
        bins, bin_center, _ = make_default_cta_binning(e_min=e_min, e_max=e_max)
    else:
        bins, bin_center, _ = make_default_cta_binning(e_min=e_min, e_max=e_max, centering='lin')


    b_68, bin_edges, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 68), bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    bins_y = np.logspace(np.log10(0.005), np.log10(50.8), 100)
    bins_y_lin = np.linspace(0.005, 0.5, 100)


    log_emin, log_emax = np.log10(bins.min().value), np.log10(bins.max().value)
    log_ymin, log_ymax = np.log10(bins_y.min()), np.log10(bins_y.max())

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xscale('log')
    ax.set_ylim([0.02, 20])
    if log:
        ax.set_yscale('log')
    if hexbin:
        if log:
            im = ax.hexbin(x, y, xscale='log', yscale='log', extent=(log_emin, log_emax, log_ymin, log_ymax), cmap=default_cmap, linewidths=0.1)
        else:
            im = ax.hexbin(x, y, extent=(log_emin, log_emax, bins_y_lin.min(), bins_y_lin.max()), cmap=default_cmap, linewidths=0.1)
        add_colorbar_to_figure(im, fig, ax)

    ax.plot(bin_centers, b_68, 'b--', lw=2, color=main_color, label='68% Percentile')

    if x2 is not None and y2 is not None:
        b_68, bin_edges, _ = binned_statistic(x2, y2, statistic=lambda y2: np.nanpercentile(y2, 68), bins=bins)
        bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        ax.plot(bin_centers, b_68, 'g--', lw=2, color=second_color or 'green', label='68% Percentile Hillas')

    ax.set_ylabel(r'$\Theta \,\, / \,\, \si{\degree}$')
    ax.set_xlabel(r'$E_{\mathrm{MC}} \,\, / \,\, \si{\tera\electronvolt}$')
    #ax.legend()
    ax.set_title(name+f'\n samples: {len(y)}')
    if out_file:
        fig.savefig(out_file)
        return 0

    return fig, ax


def plot_angular_resolution_vs_multi(x, y, y2=None, name='', out_file=None, percentile=68):

    bins, bin_center, _ = make_linear_binning(x_min=min(x), x_max=max(x))

    b_68, bin_edges, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, percentile), bins=bins)

    bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    bins_y = np.logspace(np.log10(0.005), np.log10(50.8), 100)

    log_emin, log_emax = np.log10(bins.min()), np.log10(bins.max())
    log_ymin, log_ymax = np.log10(bins_y.min()), np.log10(bins_y.max())

    fig, ax = plt.subplots(1, 1, figsize=figsize)


    im = ax.hexbin(x, y, yscale='log', extent=(bins.min(), bins.max(), log_ymin, log_ymax), cmap=default_cmap, norm=PowerNorm(0.5))

    # we can do better than this!
    if y2:
        b_68_2, bin_edges_2, _ = binned_statistic(x, y2, statistic=lambda y2: np.nanpercentile(y2, percentile), bins=bins)
        ax.hexbin(x, y2, yscale='log', extent=(bins.min(), bins.max(), log_ymin, log_ymax), cmap=default_cmap, norm=PowerNorm(0.5))
        ax.plot(bin_centers, b_68_2, 'b--', lw=2, color='green', label=f'{percentile}% Percentile hillas')


    add_colorbar_to_figure(im, fig, ax)
    ax.plot(bin_centers, b_68, 'b--', lw=2, color=main_color, label=f'{percentile}% Percentile')
    ax.plot(bin_centers, 2*b_68[0]/bin_centers, '-', lw=2, color='yellow', label="1/N")

    ax.set_ylabel(r'$\Theta \,\, / \,\, \si{\degree}$')
    ax.set_xlabel('Multiplicity')
    #ax.legend()
    ax.set_title(name+f'\n samples: {len(y)}')
    if out_file:
        fig.savefig(out_file)
        return 0
    return fig, ax


def plot_angular_resolution_comp(x, y, x2, y2, name='', out_file=None, second='hillas'):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    percentiles = [25, 50, 68, 90]
    alphas = [0.8, 0.6, 0.4, 0.2]

    for percentile, alpha in zip(percentiles, alphas):
        print(percentile)
        bins, bin_center, bin_widths = make_linear_binning(x_min=min(x), x_max=max(x))
        b_68, bin_edges, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, percentile), bins=bins)
        #bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        bins_y = np.logspace(np.log10(0.005), np.log10(50.8), 100)
        ax.plot(bin_center-bin_widths/2, b_68, 'bo', lw=2, color=main_color, label=f'{percentile}% ', alpha=alpha)

        bins, bin_center, bin_widths = make_linear_binning(x_min=min(x2), x_max=max(x2))
        b_68, bin_edges, _ = binned_statistic(x2, y2, statistic=lambda y2: np.nanpercentile(y2, percentile), bins=bins)
        #bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        ax.plot(bin_center-bin_widths/2, b_68, 'bo', lw=2, color='green', label=f'{percentile}% {second}', alpha=alpha)

    ax.set_ylim([0.02, 2])
    ax.set_yscale('log')
    #ax.axhline(y=1)
    #ax.axhline(y=0.1)
    #ax.axhline(y=0.01)

    ax.set_ylabel(r'$\Theta \,\, / \,\, \si{\degree}$')
    ax.set_xlabel('Multiplicity')
    # do that duplicta eremoval stuff
    #ax.legend()
    ax.set_title(name+f'\n samples: {len(y)}')
    if out_file:
        fig.savefig(out_file)
        return 0
    return fig, ax


# get columns for this
# maybe 2dhist?
def plot_stereo_vs_multi(x, y, x2, y2, out_file=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.plot(x, y, label='stereo method')
    ax.plot(x2, y2, 'b--', lw=2, color='green', label='hillas')

    ax.set_ylabel(r'$\Theta \,\, / \,\, \si{\degree}$')
    ax.set_xlabel('Multiplicity')
    ax.legend()
    ax.set_title(f'\n samples: {len(y)}')
    if out_file:
        fig.savefig(out_file, out_file=None)
        return 0
    return fig, ax


def plot_multi_vs_energy(x, y, out_file=None):
    x_bins, x_bin_center, _ = make_default_cta_binning(e_min=min(x)*u.TeV, e_max=max(x)*u.TeV)
    y_bins = np.arange(0,30,1)
    num_tel_events = np.sum(y)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    h,xedges,yedges,im = ax.hist2d(
        x,
        y/num_tel_events,
        label='stereo method',
        bins=(x_bins.to_value(u.TeV), y_bins),
        cmap=default_cmap,
        norm=PowerNorm(0.3)
        )
    ax.set_ylabel('Multiplicity')
    ax.set_xlabel('MC Energy')
    ax.set_xscale('log')
    ax.set_title(f'Multiplicity\n samples: {len(y)}')
    add_colorbar_to_figure(im, fig, ax)
    if out_file:
        fig.savefig(out_file)
        return 0
    return fig, ax 

def plot_effective_area(df_cuts, mc_spectrum, out_path):
    bins, bin_centers, bin_widths = make_default_cta_binning(bins_per_decade=15)
    hist_mc = mc_spectrum.expected_events_for_bins(energy_bins=bins)
    hist_df, _ = np.histogram(df_cuts.mc_energy.values, bins=bins)

    invalid = hist_df > hist_mc
    hist_df[invalid] = hist_mc[invalid]

    lower_conf, upper_conf = binom_conf_interval(hist_df, hist_mc, 0.95)
    gen_area = mc_spectrum.generation_area
    lower_conf = lower_conf * gen_area
    upper_conf = upper_conf * gen_area
    area = (hist_df/hist_mc)*gen_area
    lower_error = area - lower_conf
    upper_error = upper_conf - area
    mask = area > 0


    fig, ax = plt.subplots(1, 1, figsize=figsize)
    #ax.plot([1,2,3])
    plt.errorbar(
        bin_centers.value[mask],
        area.value[mask],
        xerr = bin_widths.value[mask]/2.0,
        yerr = [lower_error.value[mask], upper_error.value[mask]],
        linestyle="")

    ax.set_title('optisch anpassen, legende adden')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(out_path)
