import pandas as pd 
from sklearn.metrics import r2_score, accuracy_score
import argparse
import numpy as np
import matplotlib.pyplot as plt


def calc_r2_acc(group):
    true = np.abs(group.true_disp)
    pred = np.abs(group.disp_prediction)
    r2_ = r2_score(true, pred)
    #avg_energy = group.mc_energy.mean()
    true = np.sign(group.true_sign)
    pred = np.sign(group.disp_prediction)
    acc = accuracy_score(true, pred)

    result_df = pd.DataFrame()
    result_df['r2'] = [r2_]
    result_df['min_energy'] = [group.mc_energy.min()]
    result_df['max_energy'] = [group.mc_energy.max()]
    result_df['mean_energy'] = [
        (result_df['min_energy']+result_df['max_energy'])/2]
    result_df['width'] = [
        (result_df['max_energy']-result_df['min_energy'])/2]
    result_df['accuracy'] = [acc]
    return result_df


# rewrite this later to have more equal bins?
# right now bins contain equal amounts of events but equal energy ranges are more useful?
# https://github.com/MaxNoe/phd_thesis/blob/master/thesis/plots/plot_disp_metrics.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('data_path', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('output_2', type=str)

    args = parser.parse_args()

    df = pd.read_hdf(args.data_path, 'df') ## was aus disp performance raus kommt
    df = df.dropna(how='any')
    print(df.columns)
    
    energy = df.mc_energy
    true_disp = np.abs(df.true_disp)
    disp = np.abs(df.disp_prediction)
    mean_disp = np.mean(true_disp)
    pseudo_variance = np.abs(disp-mean_disp)
    residuals = np.abs(true_disp-disp)

    r2 = 1-np.sum(residuals**2)/np.sum(pseudo_variance**2)


    n = 10
    df = df.sort_values('mc_energy')

    df_bins = []
    for df_part in np.array_split(df, n):
        df_r = calc_r2_acc(df_part)
        df_bins.append(df_r)
    df_bins = pd.concat(df_bins)

    plt.errorbar(df_bins.mean_energy, df_bins.r2, xerr=df_bins.width, ls='')
    plt.xscale('log')
    plt.ylim(0,1)
    plt.xlabel('MC Energy [TeV]')
    plt.ylabel('Disp R^2-score')
    plt.savefig(args.output)
    plt.clf()

    df_bins = []
    for df_part in np.array_split(df, n):
        df_r = calc_r2_acc(df_part)
        df_bins.append(df_r)
    df_bins = pd.concat(df_bins)

    plt.errorbar(df_bins.mean_energy, df_bins.accuracy, xerr=df_bins.width, ls='')
    plt.xscale('log')
    plt.ylim(0,1)
    plt.xlabel('MC Energy [TeV]')
    plt.ylabel('Sign Accuracy')
    plt.savefig(args.output_2)
    plt.clf()