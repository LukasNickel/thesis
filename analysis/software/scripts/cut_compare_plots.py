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
    parser.add_argument('df_path_2', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()

    df = read_data(args.df_path, 'array_events')
    df_2 = read_data(args.df_path_2, 'array_events')

    recos = [
        ('source_alt_median', 'source_az_median', 'Median of telescope predictions', 'median'),
        ('source_alt_pairwise_mean_10.0', 'source_az_pairwise_mean_10.0', 'pairwise_mean averaging of telescope predictions', 'pairwise_mean_100'),
        ('source_alt_pairwise_median_10.0', 'source_az_pairwise_median_10.0', 'pairwise_median averaging of telescope predictions', 'pairwise_median_100'),
        ('source_alt_median_all', 'source_az_median_all', 'double median', 'median_all'),
        ('alt', 'az', 'HillasReconstructor', 'hillas')]
    recos = [reco for reco in recos if reco[0] in df.columns]

    second_color='red'

    
    for reco in recos:
        ## this removes all events hillas failed on!!!! these might still work with other methods!
        df_ = df[['num_triggered_telescopes', 'mc_energy', 'mc_alt', 'mc_az', reco[0], reco[1]]].dropna(how='any')  
        theta = angular_separation(
            df_['mc_az'].values * u.deg,
            df_['mc_alt'].values * u.deg,
            df_[reco[1]].values * u.deg,
            df_[reco[0]].values * u.deg).to(u.deg)

        df_2_ = df_2[['num_triggered_telescopes', 'mc_energy', 'mc_alt', 'mc_az', reco[0], reco[1]]].dropna(how='any')  
        theta_2 = angular_separation(
            df_2_['mc_az'].values * u.deg,
            df_2_['mc_alt'].values * u.deg,
            df_2_[reco[1]].values * u.deg,
            df_2_[reco[0]].values * u.deg).to(u.deg)
        
        plot_angular_resolution(
            x=df_['mc_energy'],
            y=theta,
            x2=df_2_['mc_energy'],
            y2=theta_2,
            name='',
            out_file=args.output_folder+'/'+reco[3]+'_cut_compare.pdf',
            log=True,
            hexbin=False,
            second_color='red')
        plt.close('all')