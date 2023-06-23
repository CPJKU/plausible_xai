import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from methods.io_utils import pickle_load

from methods.results_utils import parse_experiment_name, used_models, explainer_map, model_map, \
    get_result_name_from_file_name, color_map, smoothing_linestyle_map

sns.set_theme(style="whitegrid")
stdexpl_symbol_map = {
    'lime': '<',
    'ig': 'H',
    'gbp': '*',
    'saliency': '^'
}

architecture_symbol_map = {
    'alexnet': 'o',
    'vgg': 'v',
    'resnet': 's',
    'densenet': 'd',
    'inceptionv3': 'P'
}


def prep_results(experiment_dir):
    """ Prepares results by reading sub-directories and collecting everything in single dict. """
    available_experiments = experiment_dir.glob('./*/label_flips*pkl')

    formatted_results = {0: {}, 1: {}}

    for model_name in used_models:
        formatted_results[0][model_name] = {}
        formatted_results[1][model_name] = {}
    for result_dir in available_experiments:
        results = pickle_load(result_dir)
        model_name, explainer, _, segmented = parse_experiment_name(result_dir.parent.name)
        result_name = get_result_name_from_file_name(explainer, result_dir.name, segmented)
        formatted_results[int(segmented)][model_name][result_name] = list(results.values())

    return formatted_results


def plot_std_expl_vs_baseline(model_name, segmentation, explainers, x_slic, x_rectangles, all_results):
    """ Creates plot showing label-flip rates for varying changed segments (standard explainers vs magnitude baseline). """
    df = pd.DataFrame(all_results[segmentation][model_name])

    start_idx = 0
    x = x_rectangles
    if segmentation == 1:
        start_idx = 1
        x = x_slic

    fig, axes = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    fig.suptitle(model_name)
    axes.plot(x, df['magnitude_mean'][start_idx:], c='tab:orange', marker='x',
              label='Baseline')

    for expl in explainers:
        try:
            axes.plot(x, df[expl][start_idx:],
                      linestyle=smoothing_linestyle_map(expl.split('_')[1]),
                      marker=stdexpl_symbol_map[expl.split('_')[0]],
                      color=color_map[expl.split('_')[0]],
                      label=explainer_map[expl]
                      )
        except:
            print('{}/{}: could not plot {}'.format(model_name, segmentation, expl))

    axes.set_ylim(0.00, 1.05)
    if segmentation:
        axes.set_xlabel('% segments')
    else:
        axes.set_xlabel('# segments (k)')
    axes.set_ylabel('label-flip rate')
    plt.legend()
    plt.show()


def plot_rectangles_vs_slic(model_name, explainer, aggregation, x_slic, x_rectangles, all_results):
    """ Creates plot showing label-flip rates for different amount of changed segments (rectangular ). """
    df0 = pd.DataFrame(all_results[0][model_name])
    df1 = pd.DataFrame(all_results[1][model_name])

    fig, axes = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    fig.suptitle(explainer)
    axes.plot(np.array(list(x_rectangles))[2:] / 16, df0['{}_{}'.format(explainer, aggregation)][2:], c='tab:blue',
              marker='s', label='Rectangles')
    axes.plot(x_slic, df1['{}_{}'.format(explainer, aggregation)][1:], c='tab:green', marker='x', label='SLIC')

    axes.set_xlabel('% segments')
    axes.set_ylabel('label-flip rate')

    plt.legend()
    plt.show()


def plot_architectures(segmentation, explainer, aggregation, x_slic, x_rectangles, all_results):
    """ Creates plot showing label-flip rates for different architectures. """
    df = pd.DataFrame(all_results[segmentation])

    start_idx = 0
    x = x_rectangles
    if segmentation == 1:
        start_idx = 1
        x = x_slic

    fig, axes = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    fig.suptitle('{} ({})'.format(explainer, 'rectangles' if segmentation == 0 else 'SLIC'))
    for model in used_models:
        axes.plot(x, df[model]['{}_{}'.format(explainer, aggregation)][start_idx:],
                  linestyle=smoothing_linestyle_map(aggregation),
                  marker=architecture_symbol_map[model],
                  label=model_map[model]
                  )

    if segmentation:
        axes.set_xlabel('% segments')
    else:
        axes.set_xlabel('# segments (k)')
    axes.set_ylabel('label-flip rate')
    plt.legend()
    plt.show()


def main():
    x_rectangles = range(1, 17)
    x_slic = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # directory pointing to results of experiments - change if necessary
    experiment_dir = Path(__file__).parent.parent / 'results/experiment6_23/10_perc'
    all_results = prep_results(experiment_dir)

    aggregation = 'mean_abs'

    try:
        # plot figure 5: comparison of rectangles vs SLIC
        # first: lime
        plot_rectangles_vs_slic('resnet', 'lime', 'mean', x_slic, x_rectangles, all_results)
        # second: integrated gradient
        plot_rectangles_vs_slic('resnet', 'ig', aggregation, x_slic, x_rectangles, all_results)
        # third: saliency
        plot_rectangles_vs_slic('resnet', 'saliency', aggregation, x_slic, x_rectangles, all_results)

        # plot 6: comparison of standard explainers vs baseline
        explainers = ['lime_mean', 'ig_{}'.format(aggregation), 'saliency_{}'.format(aggregation),
                      'gbp_{}'.format(aggregation)]
        # first: alexnet
        plot_std_expl_vs_baseline('alexnet', 1, explainers, x_slic, x_rectangles, all_results)
        # second: inceptionv3
        plot_std_expl_vs_baseline('inceptionv3', 1, explainers, x_slic, x_rectangles, all_results)
        # third: densenet
        plot_std_expl_vs_baseline('densenet', 1, explainers, x_slic, x_rectangles, all_results)

        # plot 7: comparison of different architectures
        # first: lime / rectangles
        plot_architectures(0, 'lime', 'mean', x_slic, x_rectangles, all_results)
        # second: gbp / slic
        plot_architectures(1, 'gbp', aggregation, x_slic, x_rectangles, all_results)
        # third: sal / slic
        plot_architectures(1, 'saliency', aggregation, x_slic, x_rectangles, all_results)
    except KeyError:
        raise FileNotFoundError('Some results appear to be missing - '
                                'make sure to run all required experiments before calling this script!')


if __name__ == '__main__':
    main()
