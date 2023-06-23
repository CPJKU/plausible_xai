import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torchvision import datasets
from argparse import ArgumentParser

from methods.io_utils import pickle_dump, json_dump
from methods.data import get_adv_imagenet_loader
from methods.explainer_utils import get_explainer
from methods.attribution import attribute_image_features
from methods.model_utils import get_net, get_prediction, softmax
from methods.segments import adding_k_segments, get_segment_weights, create_new_adversary_with_segments


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Runs everything necessary for first set of experiments, in which label-flip rates are computed.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data-path', metavar='DIR', type=str, help='Path to image-net data.', required=True)
    parser.add_argument('--adv-path', metavar='DIR', type=str, help='Path to pre-computed adversaries.',
                        default=str(Path(__file__).parent.parent / 'adversaries/'))
    parser.add_argument('--arch', type=str, default='alexnet',
                        help='Which network to use. One of [alexnet, vgg, resnet, densenet, inceptionv3].')
    parser.add_argument('--explainer', type=str, default='lime', help='Explainer we want to investigate.')
    parser.add_argument('--segment', action='store_true', help='Use image segmentation instead of rectangular segments.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for the data_loaders')
    parser.add_argument('--subset_percentage', type=int, default=10, help='How much data to use for experiments.')
    parser.add_argument('--baseline', action='store_true', help='Also computes (magnitude) baseline.')

    return parser


def adding_fixed_k(net, adv_loader, explainer_type, k, aggregation_methods, device):
    """ Computes explanations for adversaries, and checks whether with (fixed) k most important segments we can still flip prediction of (clean) files. """
    label_flips, label_still_adversarial = {}, {}
    attrs_dict = {}

    # we count label flips for all aggregation methods of interest, so first initialise this
    for aggregation in aggregation_methods:
        label_flips[aggregation] = 0
        label_still_adversarial[aggregation] = 0

    for i, (orig_x, adv_x, mask) in tqdm(enumerate(adv_loader)):
        # get prediction of adversary
        orig_x, adv_x = orig_x.to(device), adv_x.to(device)
        adv_out = net(adv_x)
        adv_probs = softmax(adv_out)
        adv_prediction = adv_probs.argmax(dim=1)
        # prepare mask / explainer for current samples
        explainer, explainer_args = get_explainer(explainer_type, mask, device, net, lime_similarity_function=None)
        attrs = attribute_image_features(net, explainer, adv_x, adv_prediction, **explainer_args)

        # save for later computations
        attrs_dict.update({i: attrs.detach().cpu()})

        for aggregation in aggregation_methods:
            # get 'k' highest segments
            segment_weights = get_segment_weights(attrs, mask, aggregation=aggregation)
            # for all not existing segments (for image segmentation), above function returns a nan
            # so change this to something very low
            segment_weights = np.nan_to_num(segment_weights, nan=-100000)
            top_segments = np.argsort(segment_weights)[:, -k:]  # top k in reverse order
            new_adv = create_new_adversary_with_segments(orig_x, adv_x, mask, top_segments)
            # get new prediction of network (and original one)
            new_pred, _ = get_prediction(net, new_adv)
            orig_pred, _ = get_prediction(net, orig_x)

            label_flips[aggregation] += (orig_pred != new_pred).sum()
            label_still_adversarial[aggregation] += (new_pred == adv_prediction).sum()

    return attrs_dict, label_flips, label_still_adversarial


def run_experiment(data_path, adv_path, explainer_type, segment, results_path, arch, batch_size, subset_perc, baseline):
    """ Performs experiments in which label flips when choosing segments based on explainer / magnitude are investigated. """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_net(arch, device)
    k = 3
    # prepare data
    adv_loader = get_adv_imagenet_loader(adv_path, datasets.ImageNet(data_path, 'val').samples, batch_size=batch_size,
                                         subset_percentage=subset_perc, inception=arch == 'inceptionv3', segment=segment)
    n_examples = len(adv_loader.dataset)
    print("Size of validation set: {}".format(n_examples))

    agg_methods = ['mean']
    if explainer_type != 'lime':
        agg_methods.append('mean_pos')
        agg_methods.append('mean_abs')

    # Experiment section 6.2 - fixed k
    attrs_dict, label_flips, label_still_adversarial = adding_fixed_k(net, adv_loader, explainer_type, k,
                                                                      agg_methods, device)
    # show number of label flips with k segments, also show how much adversarial labels remained the same
    print('% of label flips for {} segments: {} (abs {})'.format(k, label_flips['mean'] / n_examples, label_flips['mean']))
    print('% of adversarial labels remaining: {} (abs {})'.format(label_still_adversarial['mean'] / n_examples,
                                                                  label_still_adversarial['mean']))
    # save results
    pickle_dump({
        'label_flips': label_flips,
        'n_examples': n_examples,
        'label_still_adversarial': label_still_adversarial
    }, results_path / 'experiment6_2_k={}.pkl'.format(k))

    # Experiment section 6.3 - compute label flips for different k's
    if segment:
        k_subset = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        print('computing label flips for different k percentages...')
    else:
        k_subset = range(1, 17)
        print('computing label flips for all k subsets...')

    # compute label-flip rates for various different ks (or percentages) and different aggregation methods
    for aggregation in agg_methods:
        label_flips_for_k_explainer = {
            k: adding_k_segments(net, k, adv_loader, device, attrs_dict, agg_method=aggregation, use_mag=False)
            for k in k_subset}

        print("label_flips_for_k_explainer", label_flips_for_k_explainer)
        pickle_dump(label_flips_for_k_explainer, results_path / 'label_flips_k_explainer_{}.pkl'.format(aggregation))
    # finally, compute baselines (if necessary/desired)
    if explainer_type == 'lime':
        compute_pos_exps(adv_loader, attrs_dict, device, n_examples, net, results_path)
    if baseline:
        compute_mag_exps(adv_loader, k_subset, net, results_path, device)


def compute_mag_exps(adv_loader, k_subset, net, results_path, device):
    """ Computes baseline, where magnitude of perturbation is used instead of attributions to determine importance of segments. """
    # compute label-flip rate when adding k adversarial segments based on highest magnitude thereof
    label_flips_for_k_mag = {k: adding_k_segments(net, k, adv_loader, device, use_mag=True, agg_method='magnitude')
                             for k in k_subset}
    print("label_flips_for_k_mag", label_flips_for_k_mag)
    pickle_dump(label_flips_for_k_mag, results_path / 'label_flips_k_magnitude.pkl')
    # do the same but for mean magnitude of perturbation in segments
    label_flips_for_k_mag_mean = {k: adding_k_segments(net, k, adv_loader, device, use_mag=True,
                                                       agg_method='magnitude_mean')
                                  for k in k_subset}
    print("label_flips_for_k_mag_mean", label_flips_for_k_mag_mean)
    pickle_dump(label_flips_for_k_mag_mean, results_path / 'label_flips_k_magnitude_mean.pkl')


def compute_pos_exps(adv_loader, attrs_dict, device, n_examples, net, results_path):
    """ Computes label-flip rate if we add all positively weighted segments (for lime), and same amount of segments for magnitude baseline. """
    # check flips for positive segments only
    lf_pos_lime, positive_k_lime, col_seg_weights = adding_k_segments(net, 'positive', adv_loader, device,
                                                                      attrs_dict, agg_method=None, use_mag=False)
    # look at magnitude baseline for same amount of changed segments
    lf_pos_mag, positive_k_mag, _ = adding_k_segments(net, 'positive', adv_loader, device, attrs_dict,
                                                      agg_method='magnitude', use_mag=True)
    lf_pos_mag_mean, positive_k_mag_mean, _ = adding_k_segments(net, 'positive', adv_loader, device, attrs_dict,
                                                                agg_method='magnitude_mean', use_mag=True)
    # print results, then store them in pickle file
    print('% of label flips for positive segments (lime): {}'.format(lf_pos_lime))
    print('% of label flips for positive segments (magnitude): {}'.format(lf_pos_mag))
    print('% of label flips for positive segments (magnitude_mean): {}'.format(lf_pos_mag_mean))
    pickle_dump({
        'label_flips_for_pos_lime': lf_pos_lime,
        'n_examples': n_examples,
        'label_flips_for_pos_mag': lf_pos_mag,
        'label_flips_for_pos_mag_mean': lf_pos_mag_mean,
        'positive_k_lime': positive_k_lime,
        'positive_k_mag': positive_k_mag,
        'positive_k_mag_mean': positive_k_mag_mean,
        'segment_weights': col_seg_weights
    }, results_path / 'experiment1c_k=positive.pkl')


def prep_paths(opts):
    """ Prepare all necessary paths pointing to data / directories for saving stuff. """
    data_path = Path(opts.data_path)
    adv_path = Path(opts.adv_path) / opts.arch
    print('adversarial path', adv_path)
    if not data_path.exists():
        raise NotADirectoryError('Please define a valid data-path!')
    if not adv_path.exists():
        raise NotADirectoryError('Please define a valid adversarial path!')
    experiment_name = 'experiment6_23'
    results_path = Path(__file__).parent.parent / 'results' / experiment_name
    results_path = results_path / '{}_perc'.format(opts.subset_percentage)
    explainer_result_name = opts.explainer
    if opts.segment:
        results_path = results_path / '{}_{}_sub{}_segment'.format(opts.arch, explainer_result_name,
                                                                   opts.subset_percentage)
    else:
        results_path = results_path / '{}_{}_sub{}'.format(opts.arch, explainer_result_name, opts.subset_percentage)
    print("storing results in {}".format(results_path))
    if not results_path.exists():
        results_path.mkdir(parents=True)
    return adv_path, data_path, results_path


def main():
    # parse arguments
    parser = opts_parser()
    opts = parser.parse_args()

    adv_path, data_path, results_path = prep_paths(opts)
    json_dump(vars(opts), results_path / 'opts.json')

    # run experiments
    run_experiment(data_path, adv_path, opts.explainer, opts.segment, results_path, opts.arch,
                   opts.batch_size, opts.subset_percentage, opts.baseline)


if __name__ == '__main__':
    main()
