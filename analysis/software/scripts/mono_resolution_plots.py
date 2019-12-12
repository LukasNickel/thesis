from plot_functions import plot_angular_resolution
from aict_tools.io import read_data
from astropy.coordinates.angle_utilities import angular_separation
from astropy import units as u
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('df_path', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()

    df_array = read_data(args.df_path, 'array_events')
    df_tel = read_data(args.df_path, 'telescope_events')
    df_tel = df_tel.merge(df_array[['run_id', 'array_event_id', 'mc_energy', 'mc_alt', 'mc_az', 'num_triggered_telescopes']], on=['run_id', 'array_event_id'], how='left')
    
    theta = angular_separation(
            df_tel['mc_az'].values * u.deg,
            df_tel['mc_alt'].values * u.deg,
            df_tel['source_az_prediction'].values * u.deg,
            df_tel['source_alt_prediction'].values * u.deg).to(u.deg)
    df_tel['tel_theta'] = theta

    for tel_id in [1,2,3]:
        plot_angular_resolution(
            df_tel[df_tel['telescope_type_id']==tel_id]['mc_energy'].values,
            df_tel[df_tel['telescope_type_id']==tel_id]['tel_theta'].values,
            name='',
            out_file=args.output_folder+f'/tel_vs_energy_{tel_id}.pdf',
            log=True)

    plot_angular_resolution(df_tel['mc_energy'], theta, name='Telescope prediction', out_file=args.output_folder+'/'+'tel'+'_vs_energy.pdf', log=True)
