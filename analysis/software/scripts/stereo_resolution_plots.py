from plot_functions import (
    plot_angular_resolution,
    plot_angular_resolution_comp,
    plot_angular_resolution_vs_multi,
    plot_multi_vs_energy)
from aict_tools.io import read_data
from astropy.coordinates.angle_utilities import angular_separation
from astropy import units as u
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('df_path', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()

    df = read_data(args.df_path, 'array_events')
    recos = [
        ('source_alt_median', 'source_az_median', 'Median of telescope predictions', 'median'),
        ('source_alt_pairwise_mean_10.0', 'source_az_pairwise_mean_10.0', 'pairwise_mean averaging of telescope predictions', 'pairwise_mean_100'),
        ('source_alt_pairwise_median_10.0', 'source_az_pairwise_median_10.0', 'pairwise_median averaging of telescope predictions', 'pairwise_median_100'),]
    recos = [reco for reco in recos if reco[0] in df.columns]
    
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
        plot_angular_resolution(df_['mc_energy'], theta, name='', x2=df_hillas['mc_energy'], y2=theta_hillas, out_file=args.output_folder+'/'+reco[3]+'_vs_energy.pdf', log=True)
        plt.clf()
        mask = df_['num_triggered_telescopes'] < 20
        plot_angular_resolution_vs_multi(
            df_['num_triggered_telescopes'][mask],
            theta[mask],
            name='',
            out_file=args.output_folder+'/'+reco[3]+'_vs_multi.pdf')
        plt.clf()
        plot_angular_resolution_comp(
            x=df_['num_triggered_telescopes'][mask],
            y=theta[mask],
            x2 = df_hillas['num_triggered_telescopes'][mask],
            y2=theta_hillas[mask],
            name='',
            out_file=args.output_folder+'/'+reco[3]+'_vs_multi_comp.pdf')
        plt.close('all')

    plot_multi_vs_energy(df['num_triggered_telescopes'].values, df['mc_energy'].values, out_file=args.output_folder+'/multiplicity.pdf')
