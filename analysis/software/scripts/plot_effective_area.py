import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import binom_conf_interval
import astropy.units as u
import pandas as pd
from plot_functions import plot_effective_area
import argparse
from aict_tools.io import read_data
from cta_plots.spectrum import MCSpectrum
from aict_tools.cta_helpers import horizontal_to_camera_cta_simtel
from aict_tools.preprocessing import calc_true_disp

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('df_path', type=str)
   parser.add_argument('output_path', type=str)
   parser.add_argument('-correct_signs', action='store_true')
   args = parser.parse_args()
   df_path = args.df_path
   output_path = args.output_path

   events = read_data(df_path, 'array_events')
   runs = read_data(df_path, 'runs')
   mc_spectrum = MCSpectrum.from_cta_runs(runs)

   #### implement cuts if needed 
   if args.correct_signs:
       true_x, true_y = horizontal_to_camera_cta_simtel(events)
       true_disp, true_sign = calc_true_disp(
           true_x,
           true_y,
           events['x'],
           events['y'],
           np.deg2rad(df_tel['psi']))
       predicted_sign_mask = (events['disp_prediction'] > 0)
       true_sign_mask = (true_sign > 0)
       correct_mask = (predicted_sign_mask == true_sign_mask)
       events = events[correct_mask]

   plot_effective_area(events, mc_spectrum, output_path)
