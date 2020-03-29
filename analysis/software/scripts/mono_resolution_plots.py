from plot_functions import plot_angular_resolution
from aict_tools.io import read_data, read_telescope_data
from aict_tools.configuration import AICTConfig
from aict_tools.preprocessing import calc_true_disp
from aict_tools.cta_helpers import horizontal_to_camera_cta_simtel
from astropy.coordinates.angle_utilities import angular_separation
from astropy import units as u
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('df_path', type=str)
    parser.add_argument('config_path', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()

    #df_array = read_data(args.df_path, 'array_events')
    #df_tel = read_data(args.df_path, 'telescope_events')
    #df_tel = df_tel.merge(df_array[['run_id', 'array_event_id', 'mc_energy', 'mc_alt', 'mc_az', 'num_triggered_telescopes']], on=['run_id', 'array_event_id'], how='left')
    
    config = AICTConfig.from_yaml(args.config_path)
    df_tel = read_telescope_data(args.df_path, config)
    theta = angular_separation(
            df_tel['mc_az'].values * u.deg,
            df_tel['mc_alt'].values * u.deg,
            df_tel['source_az_prediction'].values * u.deg,
            df_tel['source_alt_prediction'].values * u.deg).to(u.deg)
    df_tel['tel_theta'] = theta

    true_x, true_y = horizontal_to_camera_cta_simtel(df_tel)
    true_disp, true_sign = calc_true_disp(
        true_x,
        true_y,
        df_tel['x'],
        df_tel['y'],
        np.deg2rad(df_tel['psi']))
    predicted_sign_mask = (df_tel['disp_prediction'] > 0)
    true_sign_mask = (true_sign > 0)
    correct_mask = (predicted_sign_mask == true_sign_mask)

    for tel_id in df_tel['telescope_type_id'].unique():
        filename = args.output_folder+f'/tel_vs_energy_{tel_id}.pdf'
        plot_angular_resolution(
            df_tel[df_tel['telescope_type_id']==tel_id]['mc_energy'].values,
            df_tel[df_tel['telescope_type_id']==tel_id]['tel_theta'].values,
            name='',
            out_file=filename,
            log=True)

    plot_angular_resolution(df_tel['mc_energy'], theta, name='', out_file=args.output_folder+'/'+'tel'+'_vs_energy.pdf', log=True)

    print(df_tel['tel_theta'].describe())
    print(df_tel['tel_theta'][correct_mask].describe())
    plot_angular_resolution(
        df_tel['mc_energy'][correct_mask],
        theta[correct_mask],
        name='',
        out_file=args.output_folder+'/tel_vs_energy_correct_signs.pdf', log=True)
