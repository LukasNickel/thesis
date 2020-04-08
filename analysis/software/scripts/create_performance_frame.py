from aict_tools.io import read_telescope_data, write_data
from aict_tools.configuration import AICTConfig
import click
import pandas as pd

@click.command()
@click.argument('df_path', type=click.Path(exists=True))
@click.argument('model_type')
@click.argument('output_path', type=click.Path(exists=False))
def main(df_path, model_type, output_path):
    config = AICTConfig.from_yaml('config/aict_tel_config.yaml')
    df = read_telescope_data(df_path, config, columns=['mc_energy', 'gamma_energy_prediction', 'gamma_prediction'])
    df['gamma'] = 1
    df_perf = pd.DataFrame()
    if model_type == 'separator':
        df_perf['label_prediction'] = df.gamma_prediction
        df_perf['label'] = 1
        df_perf['probabilities'] = 0
    elif model_type =='energy':
       df_perf['label'] = df.mc_energy
       df_perf['label_prediction'] = df.gamma_energy_prediction

    df_perf['cv_fold'] = 1
    write_data(df_perf, output_path, mode='w')


if __name__ == '__main__':
    main()
