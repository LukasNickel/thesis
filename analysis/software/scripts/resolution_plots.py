from aict_tools.io import read_telescope_data, append_column_to_hdf5
import pandas as pd
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
matplotlib.use('pdf')


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
    bin_edges = np.linspace(x_min, x_max, x_max-x_min)
    idx = np.searchsorted(bin_edges, [x_min, x_max])
    max_idx = min(idx[1] + 1, len(bin_edges) - 1)
    bin_edges = bin_edges[idx[0]:max_idx]

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_centers, bin_widths


def plot_angular_resolution(x, y, name, x2=None, y2=None, log=False, out_file=None):
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

    fig, ax = plt.subplots(1, 1)

    if log:
        im = ax.hexbin(x, y, xscale='log', yscale='log', extent=(log_emin, log_emax, log_ymin, log_ymax), cmap=default_cmap, norm=PowerNorm(0.5))
    else:
        im = ax.hexbin(x, y, extent=(log_emin, log_emax, bins_y_lin.min(), bins_y_lin.max()), cmap=default_cmap, norm=PowerNorm(0.5))

    add_colorbar_to_figure(im, fig, ax)
    ax.plot(bin_centers, b_68, 'b--', lw=2, color=main_color, label='68% Percentile')

    if x2 is not None and y2 is not None:
        b_68, bin_edges, _ = binned_statistic(x2, y2, statistic=lambda y2: np.nanpercentile(y2, 68), bins=bins)
        bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        ax.plot(bin_centers, b_68, 'g--', lw=2, color='green', label='68% Percentile Hillas')

    ax.set_xscale('log')
    if log:
        ax.set_yscale('log')
    ax.axhline(y=1)
    ax.axhline(y=0.1)
    ax.axhline(y=0.01)

    ax.set_ylabel('Distance to True Position / degree')
    ax.set_xlabel(r'$E_{MC} / TeV$')
    ax.legend()
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

    fig, ax = plt.subplots(1, 1)


    im = ax.hexbin(x, y, yscale='log', extent=(bins.min(), bins.max(), log_ymin, log_ymax), cmap=default_cmap, norm=PowerNorm(0.5))

    # we can do better than this!
    if y2:
        b_68_2, bin_edges_2, _ = binned_statistic(x, y2, statistic=lambda y2: np.nanpercentile(y2, percentile), bins=bins)
        ax.hexbin(x, y2, yscale='log', extent=(bins.min(), bins.max(), log_ymin, log_ymax), cmap=default_cmap, norm=PowerNorm(0.5))
        ax.plot(bin_centers, b_68_2, 'b--', lw=2, color='green', label=f'{percentile}% Percentile hillas')


    add_colorbar_to_figure(im, fig, ax)
    ax.plot(bin_centers, b_68, 'b--', lw=2, color=main_color, label=f'{percentile}% Percentile')

    #ax.set_yscale('log')
    ax.axhline(y=1)
    ax.axhline(y=0.1)
    ax.axhline(y=0.01)

    ax.set_ylabel('Distance to True Position / degree')
    ax.set_xlabel('Multiplicity')
    ax.legend()
    ax.set_title(name+f'\n samples: {len(y)}')
    if out_file:
        fig.savefig(out_file)
        return 0
    return fig, ax


def plot_angular_resolution_comp(x, y, x2, y2, name='', out_file=None):
    fig, ax = plt.subplots(1, 1)
    percentiles = [25, 50, 68, 90]
    alphas = [0.8, 0.6, 0.4, 0.2]

    for percentile, alpha in zip(percentiles, alphas):
        bins, bin_center, _ = make_linear_binning(x_min=min(x), x_max=max(x))
        b_68, bin_edges, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, percentile), bins=bins)
        bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        bins_y = np.logspace(np.log10(0.005), np.log10(50.8), 100)
        ax.plot(bin_centers, b_68, 'bo', lw=2, color=main_color, label=f'{percentile}% ', alpha=alpha)

        bins, bin_center, _ = make_linear_binning(x_min=min(x2), x_max=max(x2))
        b_68, bin_edges, _ = binned_statistic(x2, y2, statistic=lambda y2: np.nanpercentile(y2, percentile), bins=bins)
        bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        ax.plot(bin_centers, b_68, 'bo', lw=2, color='green', label=f'{percentile}% hillas', alpha=alpha)

    ax.set_yscale('log')
    ax.axhline(y=1)
    ax.axhline(y=0.1)
    ax.axhline(y=0.01)

    ax.set_ylabel('Distance to True Position / degree')
    ax.set_xlabel('Multiplicity')
    # do that duplicta eremoval stuff
    ax.legend()
    ax.set_title(name+f'\n samples: {len(y)}')
    if out_file:
        fig.savefig(out_file)
        return 0
    return fig, ax


# get columns for this
# maybe 2dhist?
def plot_stereo_vs_multi(x, y, x2, y2, out_file=None):
    fig, ax = plt.subplots(1, 1)
    im = ax.plot(x, y, label='stereo method')
    ax.plot(x2, y2, 'b--', lw=2, color='green', label='hillas')

    ax.set_ylabel('Distance to True Position / degree')
    ax.set_xlabel('Multiplicity')
    ax.legend()
    ax.set_title(name+f'\n samples: {len(y)}')
    if out_file:
        fig.savefig(out_file, out_file=None)
        return 0
    return fig, ax


def plot_multi_vs_energy(x, y, out_file=None):
    fig, ax = plt.subplots(1, 1)
    im = ax.hist2d(x, y, label='stereo method')

    ax.set_ylabel('Multiplicity')
    ax.set_xlabel('MC Energy')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title(f'Multiplicity\n samples: {len(y)}')
    if out_file:
        fig.savefig(out_file)
        return 0
    return fig, ax 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('df_path', type=str)
    parser.add_argument('output_folder', type=str)
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    config = AICTConfig.from_yaml(args.config_path)
    model_config = config.disp

#    df = pd.read_hdf(args.df_path, 'array_events')
    df = read_data(args.df_path, 'array_events')

    recos = [
        ('source_alt_median', 'source_az_median', 'Median of telescope predictions', 'median'),
        ('source_alt_pairwise_10.0', 'source_az_pairwise_10.0', 'pairwise averaging of telescope predictions', 'pairwise_100'),
        ('source_alt_pairwise_mean_10.0', 'source_az_pairwise_mean_10.0', 'pairwise_mean averaging of telescope predictions', 'pairwise_mean_100'),
        ('source_alt_pairwise_median_10.0', 'source_az_pairwise_median_10.0', 'pairwise_median averaging of telescope predictions', 'pairwise_median_100'),
        ('source_alt_pairwise_clipped_10.0', 'source_az_pairwise_clipped_10.0', 'pairwise_clipped averaging of telescope predictions', 'pairwise_clipped_100'),
        ('source_alt_pairwise_clipped_median_10.0', 'source_az_pairwise_clipped_median_10.0', 'pairwise_clipped_median averaging of telescope predictions', 'pairwise_clipped_median_100'),
        ('source_alt_pairwise_set_10.0', 'source_az_pairwise_set_10.0', 'pairwise_set averaging of telescope predictions', 'pairwise_set_100'),
    ]
    recos = [reco for reco in recos if reco[0] in df.columns]

    plot_multi_vs_energy(df['num_triggered_telescopes'].values, df['mc_energy'].values, out_file=args.output_folder+'/multiplicity.pdf')

    for reco in recos:
        ## this removes all events hillas failed on!!!! these might still work with other methods!
        df_ = df[['num_triggered_telescopes', 'mc_energy', 'mc_alt', 'mc_az', reco[0], reco[1]]].dropna(how='any')  
        df_hillas = df[['num_triggered_telescopes', 'mc_energy', 'alt', 'az', 'mc_alt', 'mc_az']].dropna(how='any')
        theta = angular_separation(
            df_['mc_az'].values * u.deg,
            df_['mc_alt'].values * u.deg,
            df_[reco[1]].values * u.deg,
            df_[reco[0]].values * u.deg).to(u.deg)
        theta_hillas = angular_separation(
            df_hillas['mc_az'].values * u.deg,
            df_hillas['mc_alt'].values * u.deg,
            df_hillas['az'].values * u.deg,
            df_hillas['alt'].values * u.deg).to(u.deg)
        plot_angular_resolution(df_['mc_energy'], theta, name=reco[2], x2=df_hillas['mc_energy'], y2=theta_hillas, out_file=args.output_folder+'/'+reco[3]+'_vs_energy.pdf', log=True)
        plt.clf()
        plot_angular_resolution_vs_multi(df_['num_triggered_telescopes'], theta, name=reco[2], out_file=args.output_folder+'/'+reco[3]+'_vs_multi.pdf')
        plt.clf()
        plot_angular_resolution_comp(x=df_['num_triggered_telescopes'], y=theta, x2 = df_hillas['num_triggered_telescopes'], y2=theta_hillas, name=reco[2], out_file=args.output_folder+'/'+reco[3]+'_vs_multi_comp.pdf')
        plt.close('all')

    df_tel = read_data(args.df_path, 'telescope_events')
    df_tel = df_tel.merge(df[['run_id', 'array_event_id', 'mc_energy', 'mc_alt', 'mc_az', 'num_triggered_telescopes']], on=['run_id', 'array_event_id'], how='left')
    theta = angular_separation(
            df_tel['mc_az'].values * u.deg,
            df_tel['mc_alt'].values * u.deg,
            df_tel['source_az_prediction'].values * u.deg,
            df_tel['source_alt_prediction'].values * u.deg).to(u.deg)
    df_tel['tel_theta'] = theta

    plot_angular_resolution(df_tel[df_tel['telescope_type_id']==1]['mc_energy'].values, df_tel[df_tel['telescope_type_id']==1]['tel_theta'].values, name='Telescope prediction for the LSTs', out_file=args.output_folder+'/'+'tel'+'_vs_energy_lst.pdf', log=True)
    plot_angular_resolution(df_tel[df_tel['telescope_type_id']==2]['mc_energy'].values, df_tel[df_tel['telescope_type_id']==2]['tel_theta'].values, name='Telescope prediction for the MSTs', out_file=args.output_folder+'/'+'tel'+'_vs_energy_mst.pdf', log=True)
    plot_angular_resolution(df_tel[df_tel['telescope_type_id']==3]['mc_energy'].values, df_tel[df_tel['telescope_type_id']==3]['tel_theta'].values, name='Telescope prediction for the SSTs', out_file=args.output_folder+'/'+'tel'+'_vs_energy_sst.pdf', log=True)
    plot_angular_resolution_vs_multi(df_tel['num_triggered_telescopes'], theta, name='Telescope prediction', out_file=args.output_folder+'/'+'tel'+'_vs_multi.pdf')
    plot_angular_resolution(df_tel['mc_energy'], theta, name='Telescope prediction', out_file=args.output_folder+'/'+'tel'+'_vs_energy.pdf', log=True)

