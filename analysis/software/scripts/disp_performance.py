import click
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import argparse

from aict_tools.io import (
    append_column_to_hdf5,
    read_telescope_data_chunked,
    get_column_names_in_file,
    remove_column_from_file,
    load_model,
)

from aict_tools.cta_helpers import camera_to_horizontal_cta_simtel, horizontal_to_camera_cta_simtel
from aict_tools.apply import predict_disp
from aict_tools.configuration import AICTConfig
from aict_tools.preprocessing import convert_to_float32, check_valid_rows, calc_true_disp



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('data_path', type=str)
    parser.add_argument('disp_model_path', type=str)
    parser.add_argument('sign_model_path', type=str)
    parser.add_argument('configuration_path', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()

    config = AICTConfig.from_yaml(args.configuration_path)
    model_config = config.disp
    disp_model = load_model(args.disp_model_path)
    sign_model = load_model(args.sign_model_path)

    chunked_frames = []
    chunksize = 2000
    df_generator = read_telescope_data_chunked(
        args.data_path, config, chunksize, model_config.columns_to_read_apply+['mc_energy'],
        feature_generation_config=model_config.feature_generation
    )

    for df_data, start, stop in tqdm(df_generator):
        df_data[model_config.delta_column] = np.deg2rad(df_data[model_config.delta_column])
        
        df_features = convert_to_float32(df_data[model_config.features])
        valid = check_valid_rows(df_features)
        disp_abs = disp_model.predict(df_features.loc[valid].values)
        disp_sign = sign_model.predict(df_features.loc[valid].values)
        disp = np.full(len(df_features), np.nan)
        disp[valid] = disp_abs * disp_sign
        
        disp = predict_disp(
            df_data[model_config.features], disp_model, sign_model,
            log_target=model_config.log_target,
        )
        d = df_data[['run_id', 'array_event_id', 'mc_energy']].copy()

        source_x = df_data[model_config.cog_x_column] + disp * np.cos(df_data[model_config.delta_column])
        source_y = df_data[model_config.cog_y_column] + disp * np.sin(df_data[model_config.delta_column])
        df_data['source_x_prediction'] = source_x
        df_data['source_y_prediction'] = source_y
        source_alt, source_az = camera_to_horizontal_cta_simtel(df_data)                
        d['source_alt'] = source_alt
        d['source_az'] = source_az

        source_x_2 = df_data[model_config.cog_x_column] - disp * np.cos(df_data[model_config.delta_column])
        source_y_2 = df_data[model_config.cog_y_column] - disp * np.sin(df_data[model_config.delta_column])
        df_data['source_x_prediction_2'] = source_x_2
        df_data['source_y_prediction_2'] = source_y_2
        source_alt_2, source_az_2 = camera_to_horizontal_cta_simtel(df_data, x_key='source_x_prediction_2', y_key='source_y_prediction_2')
        d['source_alt_2'] = source_alt_2
        d['source_az_2'] = source_az_2

        d['disp_prediction'] = disp

        true_x, true_y = horizontal_to_camera_cta_simtel(df_data)
        true_disp, true_sign = calc_true_disp(
            true_x,
            true_y,            
            df_data[model_config.cog_x_column].values,
            df_data[model_config.cog_y_column].values,
            df_data[model_config.delta_column].values)
        d['true_disp'] = true_disp
        d['true_sign'] = true_sign

        chunked_frames.append(d)
    
    d = pd.concat(chunked_frames)
    d.to_hdf(args.output, 'df')
