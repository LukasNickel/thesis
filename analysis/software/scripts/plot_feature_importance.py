from aict_tools.configuration import AICTConfig
from sklearn.externals import joblib
from aict_tools.plotting import (
    plot_roc,
    plot_probabilities,
    plot_precision_recall,
    plot_feature_importances,
)
import argparse
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('model_path', type=str)
    parser.add_argument('config_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    config = AICTConfig.from_yaml(args.config_path)
    model_config = config.disp
    features = model_config.features
    #features = [f.replace('_', r'\_') for f in features]
    model = joblib.load(args.model_path)
    #from IPython import embed; embed()
    fig, ax = plt.subplots(1, 1)
    plot_feature_importances(model, features, ax=ax)
    plt.savefig(args.output_path)
