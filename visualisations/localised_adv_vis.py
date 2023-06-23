import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from methods.io_utils import pickle_load

name_map = {
    'lime': 'LIME',
    'ig_sm': 'IG-SG',
    'saliency_sm': 'Sal-SG',
    'gbp_sm': 'GBP-SG',
    'gbp_smsq': 'GBP-SG2',
    'saliency_smsq': 'Sal-SG2',
    'ig_smsq': 'IG-SG2',
    'gbp_var': 'GBP-VG',
    'saliency_var': 'Sal-VG',
    'ig_var': 'IG-VG',
    'ig': 'IG',
    'saliency': 'Sal',
    'gbp': 'GBP'
}

explainers = ['lime', 'saliency', 'ig', 'gbp', 'saliency_sm', 'saliency_smsq', 'saliency_var',
              'ig_sm', 'ig_smsq', 'ig_var', 'gbp_sm', 'gbp_smsq', 'gbp_var']


def get_exp(arch, explainer, agg, segment, perc):
    """ Returns experimental results for given architecture, explainer, aggregation, etc. """
    experiment_path = Path(__file__).parent.parent / 'results/experiment6_4' / '{}_perc'.format(perc)
    if not segment:
        exp_path = experiment_path / '{}_{}_{}_{}'.format(arch, explainer, agg, perc)
    else:
        exp_path = experiment_path / '{}_{}_{}_{}_segment'.format(arch, explainer, agg, perc)
    if exp_path.exists() and (exp_path / 'label_flips_for_partial_adversaries_135.pkl').exists():
        res_dict = pickle_load(exp_path / 'label_flips_for_partial_adversaries_135.pkl')
        return res_dict
    else:
        raise FileNotFoundError('No "label_flips_for_partial_adversaries_135.pkl" was found at this '
                                'location ({}), make sure to run the experiments first!'.format(exp_path))


def plot(explainers, segment, arch, perc):
    """ Plots (for given explainers, segmentation, architecture and data %) how many localised attack segments (out of 1/3/5) were found by explainer. """
    cm = plt.get_cmap('viridis')

    fig, axs = plt.subplots(1, 3, figsize=(40, 10), sharey=True)
    labels = explainers

    res = {explainer: get_exp(arch, explainer, 'mean_abs', segment, perc) or
                      {1: [0, 0], 3: [0, 0, 0, 0], 5: [0, 0, 0, 0, 0, 0]}
        for explainer in explainers}

    for c, k in enumerate([1, 3, 5]):
        colors = [cm(i) for i in np.linspace(0, 1, num=k + 1)]
        bottom = np.array([0. for ex in explainers])
        for i in range(k + 1):
            cur_res = np.array([res[ex][k][i] for ex in explainers]) / np.array(
                [np.array(res[ex][k]).sum() for ex in explainers])
            axs[c].bar(explainers, cur_res, width=.85, color=colors[i], bottom=bottom, label=str(i))
            bottom += cur_res

        axs[c].set_title('Modified segments: {}'.format(k), fontsize=30)
        axs[c].set_ylabel('Relative counts', fontsize=30)
        axs[c].legend(prop={'size': 25})
        axs[c].set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=30)
        axs[c].set_xticklabels([name_map[ex] for ex in labels], rotation=70, fontsize=25)
    plt.show()


def main():
    # plot Fig 8 (a): Inception v3, rectangular segments
    plot(explainers, False, 'inceptionv3', 10)
    # plot Fig 8 (b): Inception v3, SLIC segments
    plot(explainers, True, 'inceptionv3', 10)


if __name__ == '__main__':
    main()
