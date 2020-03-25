import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import binom_conf_interval
import astropy.units as u
import pandas as pd
from plot_functions import plot_effective_area
import argparse
from aict_tools.io import read_data
from cta_plots.spectrum import MCSpectrum


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('df_path', type=str)
   parser.add_argument('output_path', type=str)
   args = parser.parse_args()
   df_path = args.df_path
   output_path = args.output_path

   events = read_data(df_path, 'array_events')
   runs = read_data(df_path, 'runs')
   mc_spectrum = MCSpectrum.from_cta_runs(runs)

   #### implement cuts if needed 

   plot_effective_area(events, mc_spectrum, output_path)
