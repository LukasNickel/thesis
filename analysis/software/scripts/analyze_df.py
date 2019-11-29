import pandas as pd
from aict_tools.io import append_column_to_hdf5, HDFColumnAppender, read_telescope_data_chunked, read_telescope_data, get_column_names_in_file, remove_column_from_file, drop_prediction_column
import pandas as pd
import click
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
import numpy as np

from fact.io import write_data
from aict_tools.cta_helpers import horizontal_to_camera_cta_simtel
from aict_tools.cta_helpers import camera_to_horizontal_cta_simtel
from aict_tools.io import pickle_model, read_telescope_data, append_column_to_hdf5
from aict_tools.preprocessing import convert_to_float32, calc_true_disp
from aict_tools.feature_generation import feature_generation
from aict_tools.configuration import AICTConfig
from aict_tools.scripts.calculate_magic_stereo_disp import pairwise_nearest_disp
from aict_tools.scripts.calculate_veritas_stereo_disp import biggest_cluster_mean

import logging

from astropy.coordinates.angle_utilities import angular_separation
import astropy.units as u

from aict_tools.plotting import (
    plot_regressor_confusion,
    plot_bias_resolution,
    plot_feature_importances,
)
from sklearn import model_selection

import matplotlib.pyplot as plt
from aict_tools.configuration import AICTConfig
from aict_tools.cta_helpers import horizontal_to_camera_cta_simtel
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
import argparse

from resolution_plots import plot_angular_resolution, plot_angular_resolution_vs_multi
import argparse
from aict_tools.apply import predict_disp
from aict_tools.cta_helpers import apply_parallel


def aict_training(disp_model, sign_model, df_train):
    
    print(len(df_train))
    print(len(target_disp))
    #target_disp = df_train['true_disp']
    #target_sign = df_train['true_sign']

    kfold = model_selection.KFold(
        n_splits=model_config.n_cross_validations,
        shuffle=True,
        random_state=config.seed,)

    total = model_config.n_cross_validations
    scores_disp = []
    scores_sign = []
    
    for fold, (train, test) in enumerate(tqdm(kfold.split(df_train.values), total=total)):

        cv_x_train, cv_x_test = df_train.values[train], df_train.values[test]
        cv_disp_train, cv_disp_test = target_disp.values[train], target_disp.values[test]
        cv_sign_train, cv_sign_test = target_sign.values[train], target_sign.values[test]

        disp_regressor.fit(cv_x_train, cv_disp_train)
        sign_classifier.fit(cv_x_train, cv_sign_train)
        
        cv_disp_prediction = disp_regressor.predict(cv_x_test)
        cv_sign_prediction = sign_classifier.predict(cv_x_test)

        scores_disp.append(metrics.r2_score(cv_disp_test, cv_disp_prediction))
        scores_sign.append(metrics.accuracy_score(cv_sign_test, cv_sign_prediction))

    scores_disp = np.array(scores_disp)
    scores_sign = np.array(scores_sign)
    print('Cross validated R^2 scores for disp: {}'.format(scores_disp))
    print('Mean R^2 score from CV: {:0.4f} ± {:0.4f}'.format(
        scores_disp.mean(), scores_disp.std()))
    print('Cross validated accuracy for the sign: {}'.format(scores_sign))
    print('Mean accuracy from CV: {:0.4f} ± {:0.4f}'.format(
        scores_sign.mean(), scores_sign.std()
    ))

    np.random.seed(config.seed)
    disp_regressor.random_state = config.seed
    sign_classifier.random_state = config.seed

    disp_regressor.fit(df_train.values, target_disp.values)
    sign_classifier.fit(df_train.values, target_sign.values)
    

def debug_disp_performance(df, folder, prefix, stereo=False):
    # consistent df nutzen
    source_x, source_y = horizontal_to_camera_cta_simtel(df)
    df['source_x'] = source_x
    df['source_y'] = source_y
    true_disp, true_sign = calc_true_disp(
        source_x, source_y,
        df[model_config.cog_x_column], df[model_config.cog_y_column],
        np.deg2rad(df[model_config.delta_column]),
    )
    df['true_disp'] = true_disp
    df['true_sign'] = true_sign
    true_delta = np.arctan2((df['y']-df['source_y']), (df['x']-df['source_x']))   ### what exactly is this?
    df['true_delta'] = true_delta
       
    df_test = convert_to_float32(df[config.disp.features])
    target_disp = df['true_disp'].loc[df_test.index]
    target_sign = df['true_sign'].loc[df_test.index]
    
    #from IPython import embed; embed()
    disp_prediction = disp_regressor.predict(df_test.values)
    sign_prediction = sign_classifier.predict(df_test.values)
    df['disp_prediction'] = disp_prediction
    df['sign_prediction'] = sign_prediction
    #from IPython import embed; embed()
    
    print(f'Korrekt klassifizierte signs: {np.sum(true_sign == sign_prediction) / len(true_sign)}')
    
    source_x_prediction = df['x'] + disp_prediction * sign_prediction * np.cos(np.deg2rad(df['psi']))
    source_y_prediction = df['y'] + disp_prediction * sign_prediction  * np.sin(np.deg2rad(df['psi']))
    # warum -? scheint aber so zu passen...
    source_x_prediction_true_delta = df['x'] - disp_prediction * np.cos(df['true_delta'])
    source_y_prediction_true_delta = df['y'] - disp_prediction * np.sin(df['true_delta'])
    source_x_prediction_true_disp = df['x'] + target_disp * sign_prediction * np.cos(np.deg2rad(df['psi']))
    source_y_prediction_true_disp = df['y'] + target_disp * sign_prediction  * np.sin(np.deg2rad(df['psi']))
    source_x_prediction_true_disp_sign = df['x'] + target_disp * target_sign * np.cos(np.deg2rad(df['psi']))
    source_y_prediction_true_disp_sign = df['y'] + target_disp * target_sign  * np.sin(np.deg2rad(df['psi']))
    source_x_prediction_2 = df['x'] - disp_prediction * sign_prediction * np.cos(np.deg2rad(df['psi']))
    source_y_prediction_2 = df['y'] - disp_prediction * sign_prediction  * np.sin(np.deg2rad(df['psi']))
    
    df['source_x_pred'] = source_x_prediction
    df['source_y_pred'] = source_y_prediction
    df['source_x_pred_2'] = source_x_prediction_2
    df['source_y_pred_2'] = source_y_prediction_2
    df['source_x_pred_true_delta'] = source_x_prediction_true_delta
    df['source_y_pred_true_delta'] = source_y_prediction_true_delta
    df['source_x_pred_true_disp'] = source_x_prediction_true_disp
    df['source_y_pred_true_disp'] = source_y_prediction_true_disp    
    df['source_x_pred_true_disp_sign'] = source_x_prediction_true_disp_sign
    df['source_y_pred_true_disp_sign'] = source_y_prediction_true_disp_sign    
    
    ## psi vs true psi
    cos_psi = np.cos(np.deg2rad(df['psi']))
    cos_delta = np.cos(true_delta)
    cos_psi_sign = np.cos(np.deg2rad(df['psi'] * target_sign.values))


    plt.hist(cos_psi, bins=30, alpha=0.3, label='$\cos(\psi)$', range=[-1,1]);
    plt.hist(np.abs(cos_delta), bins=30, alpha=0.3, label='$\cos(\delta)$', range=[-1,1]);
    plt.legend()
    plt.title('Abweichung zwischen Psi und Delta')
    plt.savefig(folder+'/'+prefix+'_psi_vs_mc_1.pdf')
    plt.clf()
    
    plt.hist(cos_psi-np.abs(cos_delta), bins=30, range=[-1,1]);
    plt.title('$\cos(\psi) - |\cos(\delta)|$')
    plt.savefig(folder+'/'+prefix+'_psi_vs_mc_2.pdf')
    plt.clf()
    
    plt.hist(cos_psi+cos_delta*target_sign.values, bins=30, range=[-1,1])
    plt.title('$\cos(\psi) - \cos(\delta) * sign$')
    plt.savefig(folder+'/'+prefix+'_psi_vs_mc_2.pdf')
    plt.clf()
    
    plt.hist(disp_prediction - target_disp, bins=30, range=[-0.6,0.6]);    
    plt.title('Disp - true disp')
    plt.savefig(folder+'/'+prefix+'_disp_vs_mc.pdf')
    plt.clf()
    
    alt_pred, az_pred = camera_to_horizontal_cta_simtel(df, x_key='source_x_pred', y_key='source_y_pred')
    alt_pred_2, az_pred_2 = camera_to_horizontal_cta_simtel(df, x_key='source_x_pred_2', y_key='source_y_pred_2')
    alt_pred_true_delta, az_pred_true_delta = camera_to_horizontal_cta_simtel(df, x_key='source_x_pred_true_delta', y_key='source_y_pred_true_delta')
    alt_pred_true_disp, az_pred_true_disp = camera_to_horizontal_cta_simtel(df, x_key='source_x_pred_true_disp', y_key='source_y_pred_true_disp')
    alt_pred_true_disp_sign, az_pred_true_disp_sign = camera_to_horizontal_cta_simtel(df, x_key='source_x_pred_true_disp_sign', y_key='source_y_pred_true_disp_sign')
    
    
    df['source_alt_prediction'] = alt_pred
    df['source_az_prediction'] = az_pred
    df['source_alt_prediction_2'] = alt_pred_2
    df['source_az_prediction_2'] = az_pred_2
    df['alt_pred_true_delta'] = alt_pred_true_delta
    df['az_pred_true_delta'] = az_pred_true_delta
    df['alt_pred_true_disp'] = alt_pred_true_disp
    df['az_pred_true_disp'] = az_pred_true_disp
    df['alt_pred_true_disp_sign'] = alt_pred_true_disp_sign
    df['az_pred_true_disp_sign'] = az_pred_true_disp_sign
    
    theta = angular_separation(
        df['mc_az'].values * u.deg,
        df['mc_alt'].values * u.deg,
        df['source_az_prediction'].values * u.deg,
        df['source_alt_prediction'].values * u.deg).to(u.deg)
    theta_true_delta = angular_separation(
        df['mc_az'].values * u.deg,
        df['mc_alt'].values * u.deg,
        df['az_pred_true_delta'].values * u.deg,
        df['alt_pred_true_delta'].values * u.deg).to(u.deg)
    theta_true_disp = angular_separation(
        df['mc_az'].values * u.deg,
        df['mc_alt'].values * u.deg,
        df['az_pred_true_disp'].values * u.deg,
        df['alt_pred_true_disp'].values * u.deg).to(u.deg)
    theta_true_disp_sign = angular_separation(
        df['mc_az'].values * u.deg,
        df['mc_alt'].values * u.deg,
        df['az_pred_true_disp_sign'].values * u.deg,
        df['alt_pred_true_disp_sign'].values * u.deg).to(u.deg)
    
    print(f'Rekos unter 1°: {np.sum(theta.value**2 < 1) / len(theta)}')
    print(f'68% Theta^2: {np.percentile(theta, 68)}')    
    #from IPython import embed; embed()
    plt.hist((theta.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.percentile(theta, 68), color='k', linestyle='dashed', linewidth=1, label=np.percentile(theta, 68))
    plt.legend()
    plt.title(r'$\theta^2$')
    plt.savefig(folder+'/'+prefix+'_theta.pdf')
    plt.clf()

    non_zero = (theta.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta)} entries')
    fig, ax = plot_angular_resolution(df['mc_energy'].values[non_zero], theta[non_zero], r'$\theta^2$' + f'samples: {len(theta)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy.pdf')
    plt.clf() 

    print(f'Rekos unter 1°: {np.sum(theta_true_delta.value**2 < 1) / len(theta_true_delta)}')
    print(f'68% Theta^2 bei wahrem delta und sign: {np.percentile(theta_true_delta, 68)}')
    plt.hist((theta_true_delta.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.percentile(theta_true_delta, 68), color='k', linestyle='dashed', linewidth=1, label=np.percentile(theta_true_delta, 68))
    plt.title(r'$\theta^2$ bei wahrem Winkel')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_true_psi_sign.pdf')
    plt.clf()
    non_zero = (theta_true_delta.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta)} entries')
    fig, ax = plot_angular_resolution(df['mc_energy'].values[non_zero], theta_true_delta[non_zero], r'$\theta^2$ bei wahrem Winkel' + f'samples: {len(theta_true_delta)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_true_psi_sign.pdf')
    plt.clf()

    print(f'Rekos unter 1°: {np.sum(theta_true_disp.value**2 < 1) / len(theta_true_disp)}')
    print(f'68% Theta^2 bei wahrem disp: {np.percentile(theta_true_disp, 68)}')
    plt.hist((theta_true_disp.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.percentile(theta_true_disp, 68), color='k', linestyle='dashed', linewidth=1, label=np.percentile(theta_true_disp, 68))
    plt.title(r'$\theta^2$ bei wahrem disp')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_true_disp.pdf')
    plt.clf()
    non_zero = (theta_true_disp.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta)} entries')
    fig, ax = plot_angular_resolution(df['mc_energy'].values[non_zero], theta_true_disp[non_zero], r'$\theta^2$ bei wahrem disp' + f'samples: {len(theta_true_disp)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_true_disp.pdf')
    plt.clf()
    
    print(f'Rekos unter 1°: {np.sum(theta_true_disp_sign.value**2 < 1) / len(theta_true_disp_sign)}')
    print(f'68% Theta^2 bei wahrem disp und sign: {np.percentile(theta_true_disp_sign, 68)}')
    plt.hist((theta_true_disp_sign.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.percentile(theta_true_disp_sign, 68), color='k', linestyle='dashed', linewidth=1, label=np.percentile(theta_true_disp_sign, 68))
    plt.title(r'$\theta^2$ bei wahrem disp und sign')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_true_disp_sign.pdf')
    plt.clf()
    non_zero = (theta_true_disp_sign.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta)} entries')
    fig, ax = plot_angular_resolution(df['mc_energy'].values[non_zero], theta_true_disp_sign[non_zero], r'$\theta^2$ bei wahrem disp und sign' + f'samples: {len(theta_true_disp_sign)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_true_disp_sign.pdf')
    plt.clf()
    df.to_hdf(folder+'/df'+'_'+prefix+'.hdf5', 'w')

    if not stereo:
        return 0
    
    #from IPython import embed; embed()
    df['weights'] = 1
    df_grouped = df.groupby(['run_id', 'array_event_id'])
    array_df = apply_parallel(df_grouped, pairwise_nearest_disp, n_jobs=20, eps=0.5)
    array_df.index.names = ['run_id', 'array_event_id', None]
    array_df = array_df.droplevel(2)

    # does this work??
    array_df['mc_alt'] = df_grouped.agg('mean')['mc_alt'] 
    array_df['mc_az'] = df_grouped.agg('mean')['mc_az']
    array_df['mc_alt_unc'] = df_grouped.agg('std')['mc_alt'] 
    array_df['mc_az_unc'] = df_grouped.agg('std')['mc_az']
    array_df['mc_energy'] = df_grouped.agg('mean')['mc_energy']
    array_df['num_triggered_telescopes'] = df_grouped.agg('mean')['num_triggered_telescopes']
    array_df['alt_mean'] = df_grouped.agg('mean')['source_alt_prediction']
    array_df['az_mean'] = df_grouped.agg('mean')['source_az_prediction']
    array_df['alt_median'] = df_grouped.agg('median')['source_alt_prediction']
    array_df['az_median'] = df_grouped.agg('median')['source_az_prediction']
    from IPython import embed; embed()
    array_df.dropna(how='any', inplace=True)
    theta_magic = angular_separation(
        array_df['mc_az'].values * u.deg,
        array_df['mc_alt'].values * u.deg,
        array_df['source_az_pairwise'].values * u.deg,
        array_df['source_alt_pairwise'].values * u.deg).to(u.deg)
    theta_mean = angular_separation(
        array_df['mc_az'].values * u.deg,
        array_df['mc_alt'].values * u.deg,
        array_df['az_mean'].values * u.deg,
        array_df['alt_mean'].values * u.deg).to(u.deg)
    theta_median = angular_separation(
        array_df['mc_az'].values * u.deg,
        array_df['mc_alt'].values * u.deg,
        array_df['az_median'].values * u.deg,
        array_df['alt_median'].values * u.deg).to(u.deg)
    print(f'Rekos unter 1°: {np.sum(theta_magic.value**2 < 1) / len(theta_magic)}')
    print(f'68% Theta^2 mit magic methode: {np.nanpercentile(theta_magic, 68)}, samples: {len(theta_magic)}')
    plt.hist((theta_magic.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.nanpercentile(theta_magic, 68), color='k', linestyle='dashed', linewidth=1, label=np.nanpercentile(theta_magic, 68))
    plt.title(r'$\theta^2$ mit magic methode')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_magic.pdf')
    plt.clf()
    non_zero = (theta_magic.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta_magic)} entries')
    fig, ax = plot_angular_resolution(array_df['mc_energy'].values[non_zero], theta_magic[non_zero], r'$\theta^2$ mit magic methode' + f'samples: {len(theta_magic)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_magic.pdf')
    plt.clf()
    fig, ax = plot_angular_resolution_vs_multi(array_df['num_triggered_telescopes'].values[non_zero], theta_magic[non_zero], r'$\theta^2$ mit magic methode' + f'samples: {len(theta_magic)}')
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_magic_multi.pdf')
    plt.clf()

    print(f'Rekos unter 1°: {np.sum(theta_mean.value**2 < 1) / len(theta_mean)}')
    print(f'68% Theta^2 mit mean methode: {np.nanpercentile(theta_mean, 68)}, samples: {len(theta_mean)}')
    plt.hist((theta_mean.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.nanpercentile(theta_mean, 68), color='k', linestyle='dashed', linewidth=1, label=np.nanpercentile(theta_mean, 68))
    plt.title(r'$\theta^2$ mit mean methode')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_mean.pdf')
    plt.clf()
    non_zero = (theta_mean.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta_mean)} entries')
    fig, ax = plot_angular_resolution(array_df['mc_energy'].values[non_zero], theta_mean[non_zero], r'$\theta^2$ mit mean methode' + f'samples: {len(theta_mean)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_mean.pdf')
    plt.clf()

    print(f'Rekos unter 1°: {np.sum(theta_median.value**2 < 1) / len(theta_median)}')
    print(f'68% Theta^2 mit median methode: {np.nanpercentile(theta_median, 68)}, samples: {len(theta_median)}')
    plt.hist((theta_median.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.nanpercentile(theta_median, 68), color='k', linestyle='dashed', linewidth=1, label=np.nanpercentile(theta_median, 68))
    plt.title(r'$\theta^2$ mit median methode')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_median.pdf')
    plt.clf()
    non_zero = (theta_median.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta_median)} entries')
    fig, ax = plot_angular_resolution(array_df['mc_energy'].values[non_zero], theta_median[non_zero], r'$\theta^2$ mit median methode' + f'samples: {len(theta_median)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_median.pdf')
    plt.clf()
    fig, ax = plot_angular_resolution(array_df['num_triggered_telescopes'].values[non_zero], theta_median[non_zero], r'$\theta^2$ mit median methode' + f'samples: {len(theta_median)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_median_multi.pdf')
    plt.clf()
    ###################
    from IPython import embed; embed()
    return 0
    # only true signs:
    df_grouped = df[df['sign_prediction'] == df['true_sign']].groupby(['run_id', 'array_event_id'])
    array_df = apply_parallel(df_grouped, pairwise_nearest_disp, n_jobs=20, eps=0.5)
    array_df.index.names = ['run_id', 'array_event_id', None]
    array_df = array_df.droplevel(2)

    # does this work??
    array_df['mc_alt'] = df_grouped.agg('mean')['mc_alt'] 
    array_df['mc_az'] = df_grouped.agg('mean')['mc_az']
    array_df['mc_alt_unc'] = df_grouped.agg('std')['mc_alt'] 
    array_df['mc_az_unc'] = df_grouped.agg('std')['mc_az']
    array_df['mc_energy'] = df_grouped.agg('mean')['mc_energy']
    array_df['alt_mean'] = df_grouped.agg('mean')['source_alt_prediction']
    array_df['az_mean'] = df_grouped.agg('mean')['source_az_prediction']
    array_df['alt_median'] = df_grouped.agg('median')['source_alt_prediction']
    array_df['az_median'] = df_grouped.agg('median')['source_az_prediction']
    array_df.dropna(how='any', inplace=True)
    theta_magic_true_signs = angular_separation(
        array_df['mc_az'].values * u.deg,
        array_df['mc_alt'].values * u.deg,
        array_df['az_pairwise_disp'].values * u.deg,
        array_df['alt_pairwise_disp'].values * u.deg).to(u.deg)
    theta_mean_true_signs = angular_separation(
        array_df['mc_az'].values * u.deg,
        array_df['mc_alt'].values * u.deg,
        array_df['az_mean'].values * u.deg,
        array_df['alt_mean'].values * u.deg).to(u.deg)
    theta_median_true_signs = angular_separation(
        array_df['mc_az'].values * u.deg,
        array_df['mc_alt'].values * u.deg,
        array_df['az_median'].values * u.deg,
        array_df['alt_median'].values * u.deg).to(u.deg)
    print(f'Rekos unter 1°: {np.sum(theta_magic_true_signs.value**2 < 1) / len(theta_magic_true_signs)}')
    print(f'68% Theta^2 mit magic methode: {np.nanpercentile(theta_magic_true_signs, 68)}, samples: {len(theta_magic_true_signs)}')
    plt.hist((theta_magic_true_signs.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.nanpercentile(theta_magic_true_signs, 68), color='k', linestyle='dashed', linewidth=1, label=np.nanpercentile(theta_magic_true_signs, 68))
    plt.title(r'$\theta^2$ mit magic methode')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_magic_true_signs.pdf')
    plt.clf()
    non_zero = (theta_magic_true_signs.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta_magic_true_signs)} entries')
    fig, ax = plot_angular_resolution(array_df['mc_energy'].values[non_zero], theta_magic_true_signs[non_zero], r'$\theta^2$ mit magic methode' + f'samples: {len(theta_magic_true_signs)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_magic_true_signs.pdf')
    plt.clf()

    print(f'Rekos unter 1°: {np.sum(theta_mean_true_signs.value**2 < 1) / len(theta_mean_true_signs)}')
    print(f'68% Theta^2 mit mean methode: {np.nanpercentile(theta_mean_true_signs, 68)}, samples: {len(theta_mean_true_signs)}')
    plt.hist((theta_mean_true_signs.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.nanpercentile(theta_mean_true_signs, 68), color='k', linestyle='dashed', linewidth=1, label=np.nanpercentile(theta_mean_true_signs, 68))
    plt.title(r'$\theta^2$ mit mean methode')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_mean_true_signs.pdf')
    plt.clf()
    non_zero = (theta_mean_true_signs.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta_mean_true_signs)} entries')
    fig, ax = plot_angular_resolution(array_df['mc_energy'].values[non_zero], theta_mean_true_signs[non_zero], r'$\theta^2$ mit mean methode' + f'samples: {len(theta_mean_true_signs)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_mean_true_signs.pdf')
    plt.clf()

    print(f'Rekos unter 1°: {np.sum(theta_median_true_signs.value**2 < 1) / len(theta_median_true_signs)}')
    print(f'68% Theta^2 mit median methode: {np.nanpercentile(theta_median_true_signs, 68)}, samples: {len(theta_median_true_signs)}')
    plt.hist((theta_median_true_signs.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.nanpercentile(theta_median_true_signs, 68), color='k', linestyle='dashed', linewidth=1, label=np.nanpercentile(theta_median_true_signs, 68))
    plt.title(r'$\theta^2$ mit median methode')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_median_true_signs.pdf')
    plt.clf()
    non_zero = (theta_median_true_signs.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta_median_true_signs)} entries')
    fig, ax = plot_angular_resolution(array_df['mc_energy'].values[non_zero], theta_median_true_signs[non_zero], r'$\theta^2$ mit median methode' + f'samples: {len(theta_median_true_signs)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_median_true_signs.pdf')
    plt.clf()

    ########
    ### veritas####
    ########
    # does this work??
    df['weights'] = 1
    df_grouped = df.groupby(['run_id', 'array_event_id'])
    array_df = apply_parallel(df_grouped, biggest_cluster_mean, n_jobs=20, eps=1)
    array_df.index.names = ['run_id', 'array_event_id', None]
    array_df = array_df.droplevel(2)

    array_df['mc_alt'] = df_grouped.agg('mean')['mc_alt'] 
    array_df['mc_az'] = df_grouped.agg('mean')['mc_az']
    array_df['mc_alt_unc'] = df_grouped.agg('std')['mc_alt'] 
    array_df['mc_az_unc'] = df_grouped.agg('std')['mc_az']
    array_df['mc_energy'] = df_grouped.agg('mean')['mc_energy'] 
    array_df.dropna(how='any', inplace=True)
    theta_ver = angular_separation(
        array_df['mc_az'].values * u.deg,
        array_df['mc_alt'].values * u.deg,
        array_df['az_cluster_mean'].values * u.deg,
        array_df['alt_cluster_mean'].values * u.deg).to(u.deg)
    print(f'Rekos unter 1°: {np.sum(theta_ver.value**2 < 1) / len(theta_ver)}')
    print(f'68% Theta^2 mit cluster methode: {np.nanpercentile(theta_ver, 68)}, samples: {len(theta_ver)}')
    plt.hist((theta_ver.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.nanpercentile(theta_ver, 68), color='k', linestyle='dashed', linewidth=1, label=np.nanpercentile(theta_ver, 68))
    plt.title(r'$\theta^2$ mit cluster methode')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_cluster.pdf')
    plt.clf()
    non_zero = (theta_ver.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta_ver)} entries')
    fig, ax = plot_angular_resolution(array_df['mc_energy'].values[non_zero], theta_ver[non_zero], r'$\theta^2$ mit cluster methode' + f'samples: {len(theta_ver)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_cluster.pdf')
    plt.clf()

    ###################
    # only true signs:
    df_grouped = df[df['sign_prediction'] == df['true_sign']].groupby(['run_id', 'array_event_id'])
    array_df = apply_parallel(df_grouped, biggest_cluster_mean, n_jobs=20, eps=1)
    array_df.index.names = ['run_id', 'array_event_id', None]
    array_df = array_df.droplevel(2)

    # does this work??
    array_df['mc_alt'] = df_grouped.agg('mean')['mc_alt'] 
    array_df['mc_az'] = df_grouped.agg('mean')['mc_az']
    array_df['mc_alt_unc'] = df_grouped.agg('std')['mc_alt'] 
    array_df['mc_az_unc'] = df_grouped.agg('std')['mc_az']
    array_df['mc_energy'] = df_grouped.agg('mean')['mc_energy']
    array_df.dropna(how='any', inplace=True)
    theta_ver_true_signs = angular_separation(
        array_df['mc_az'].values * u.deg,
        array_df['mc_alt'].values * u.deg,
        array_df['az_cluster_mean'].values * u.deg,
        array_df['alt_cluster_mean'].values * u.deg).to(u.deg)
    print(f'Rekos unter 1°: {np.sum(theta_ver_true_signs.value**2 < 1) / len(theta_ver_true_signs)}')
    print(f'68% Theta^2 mit magic methode: {np.nanpercentile(theta_ver_true_signs, 68)}, samples: {len(theta_ver_true_signs)}')
    plt.hist((theta_ver_true_signs.value)**2, bins=30, range=[0,0.5]);
    plt.axvline(np.nanpercentile(theta_ver_true_signs, 68), color='k', linestyle='dashed', linewidth=1, label=np.nanpercentile(theta_ver_true_signs, 68))
    plt.title(r'$\theta^2$ mit cluster methode')
    plt.legend()
    plt.savefig(folder+'/'+prefix+'_theta_cluster_true_signs.pdf')
    plt.clf()
    non_zero = (theta_ver_true_signs.value**2 > 0)
    print(f'dropping {np.sum(~non_zero)} of {len(theta_ver_true_signs)} entries')
    fig, ax = plot_angular_resolution(array_df['mc_energy'].values[non_zero], theta_ver_true_signs[non_zero], r'$\theta^2$ mit cluster methode' + f'samples: {len(theta_ver_true_signs)}', log=True)
    plt.savefig(folder+'/'+prefix+'_theta_vs_energy_cluster_true_signs.pdf')
    plt.clf()
    from IPython import embed; embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('diffuse_file', type=str)
    parser.add_argument('gamma_file', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()

    config = AICTConfig.from_yaml('aict_tel_config.yaml')
    model_config = config.disp
    disp_regressor = model_config.disp_regressor
    sign_classifier = model_config.sign_classifier

    diff_tel = read_telescope_data(
            args.diffuse_file, config, model_config.columns_to_read_apply+['mc_energy', 'num_triggered_telescopes'],
            feature_generation_config=model_config.feature_generation)
    
    #diff_tel.replace([np.inf, -np.inf], np.nan, inplace=True)  
    diff_tel.dropna(how='any', inplace=True)
    #from IPython import embed; embed() # hier gleich groß!
    diff_lst = diff_tel#[diff_tel['telescope_type_id'] == 2]#[0:50]

    source_x, source_y = horizontal_to_camera_cta_simtel(diff_lst)
    diff_lst['source_x'] = source_x
    diff_lst['source_y'] = source_y

    true_disp, true_sign = calc_true_disp(
            source_x, source_y,
            diff_lst[model_config.cog_x_column], diff_lst[model_config.cog_y_column],
            np.deg2rad(diff_lst[model_config.delta_column]),
        )
    diff_lst[model_config.delta_column] = np.deg2rad(diff_lst[model_config.delta_column])
    diff_lst['true_disp'] = true_disp
    diff_lst['true_sign'] = true_sign
    true_delta = np.arctan2((diff_lst['y']-diff_lst['source_y']), (diff_lst['x']-diff_lst['source_x']))   ### what exactly is this?
    diff_lst['true_delta'] = true_delta
    feature_generation(diff_lst, model_config.feature_generation, inplace=True)


    df_train_test = convert_to_float32(diff_lst[config.disp.features])#.drop('mc_energy', axis=1)
    df_train, df_test = train_test_split(df_train_test, test_size=0.1)

    target_disp = diff_lst['true_disp']#.loc[df_train.index]
    target_sign = diff_lst['true_sign']#.loc[df_train.index]
    #from IPython import embed; embed()  # targets gleich!
    aict_training(disp_regressor, sign_classifier, df_train_test)
    #from IPython import embed; embed()
    #disp_prediction = disp_regressor.predict(df_train.values)
    #sign_prediction = sign_classifier.predict(df_train.values)

    features = model_config.features
   #features.remove('mc_energy')

    fig, ax = plt.subplots(1, 1)
    plot_feature_importances(disp_regressor, features, ax=ax)
    plt.savefig(args.output_folder+'/feature_importances_disp.pdf')
    plt.clf()

    fig, ax = plt.subplots(1, 1)
    plot_feature_importances(sign_classifier, features, ax=ax)
    plt.savefig(args.output_folder+'/feature_importances_sign.pdf')
    plt.clf()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    df_pointlike = read_telescope_data(
            args.gamma_file, config, model_config.columns_to_read_apply+['num_triggered_telescopes', 'mc_energy'],
            feature_generation_config=model_config.feature_generation)#[0:500]
    
    df_pointlike.replace([np.inf, -np.inf], np.nan, inplace=True)  
    df_pointlike.dropna(how='any', inplace=True)
    df_pointlike_lst = df_pointlike#[df_pointlike['telescope_type_id'] == 1]

    #from IPython import embed; embed()
    debug_disp_performance(diff_lst.loc[df_train_test.index], args.output_folder, 'train', stereo=True)
    #debug_disp_performance(diff_lst.loc[df_test.index], args.output_folder, 'test', stereo=True)
    debug_disp_performance(df_pointlike, args.output_folder, 'gamma', stereo=True)