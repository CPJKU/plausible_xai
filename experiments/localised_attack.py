import torch
import pickle
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torchvision import datasets
from argparse import ArgumentParser

from methods.data import get_adv_imagenet_loader
from methods.explainer_utils import get_explainer
from methods.model_utils import get_net, get_prediction
from methods.attribution import attribute_image_features
from methods.segments import get_segment_weights, create_new_adversary_with_segments


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Runs everything necessary for first set of experiments, in which ' \
           'predictions on adversarial examples are explained.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data-path', metavar='DIR', type=str, help='Path to image-net data.', required=True)
    parser.add_argument('--adv-path', metavar='DIR', type=str, help='Path to pre-computed adversaries.',
                        default=str(Path(__file__).parent.parent / 'adversaries/'))
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size of dataloaders.')
    parser.add_argument('--subset_percentage', type=int, default=10, help='How much data to use for experiments.')
    parser.add_argument('--arch', type=str, default='alexnet', help='Which network to use. One of [alexnet, vgg, '
                                                                    'resnet, densenet, inceptionv3].')
    parser.add_argument('--explainer', type=str, default='lime', help='Explainer we want to investigate.')
    parser.add_argument('--aggregation', type=str, default='mean', help='Method to aggregate gradients per segment.')
    parser.add_argument('--segment', action='store_true', help='Use image segmentation instead of rectangular segments.')
    parser.add_argument('--ks', nargs='+', default=[1, 3, 5], type=int,
                        help='The numbers of k segments which we want to take a look at. ')

    return parser


def get_partial_adversary(adv_x, k, mask, net, orig_x):
    """ Refines adversary to contain only k segment of the perturbation. """
    # get 'k' highest segments (according to magnitude)
    delta = torch.abs(orig_x - adv_x).detach()
    segment_mags = get_segment_weights(delta, mask, 'magnitude')
    seg_lens = np.array([len(s) for s in segment_mags])
    inds = np.argwhere(seg_lens >= k).flatten()
    top_segments = np.argsort(segment_mags)[..., -k:]
    # create new adversary
    new_adv = create_new_adversary_with_segments(orig_x, adv_x, mask, top_segments)
    # get new prediction of network
    new_pred, _ = get_prediction(net, new_adv)
    return new_adv, new_pred, top_segments, inds


def check_new_explanation(net, explainer, adv, adv_pred, explainer_args, mask, agg, k, real_segs):
    """ Computes explanation of 'new' adversary and compares top scoring segments with real ones. """
    # compute new explanation of partial adversary
    attrs = attribute_image_features(net, explainer, adv, adv_pred, **explainer_args)
    # get 'k' highest segments
    segment_weights = get_segment_weights(attrs, mask, aggregation=agg)
    top_segs = np.argsort(segment_weights)[..., -k:]
    correctly_found = [len(set(t).intersection(set(r))) for t, r in zip(top_segs, real_segs)]
    return correctly_found


def run_experiment(data_path, adv_path, explainer_type, segment, aggregation, ks,
                   arch, save_path, batch_size, subset):
    """ Runs experiment where first partial adversaries are computed, and then we check whether explainer finds correct segments. """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_net(arch, device)
    adv_loader = get_adv_imagenet_loader(adv_path, datasets.ImageNet(data_path, 'val').samples, batch_size=batch_size,
                                         subset_percentage=subset, inception=arch == 'inceptionv3', segment=segment)
    res = {k: [0 for _ in range(k + 1)] for k in ks}
    nr_of_rem = {k: 0 for k in ks}
    append_k = ''

    for k in ks:
        nr_of_rem_advs = 0
        for orig_x, adv_x, mask in tqdm(adv_loader):
            # get prediction of adversary
            orig_x, adv_x = orig_x.to(device), adv_x.to(device)
            orig_pred, _ = get_prediction(net, orig_x)

            # first, we need to get partial adversary
            new_adv, new_pred, chosen_segs, enough_segs = get_partial_adversary(adv_x, k, mask, net, orig_x)
            # check whether label is still flipped
            # (we just want new prediction to be something else than original (clean) prediction)
            nr_of_rem_advs += (new_pred[enough_segs] != orig_pred[enough_segs]).sum().item()
            _indices = np.argwhere(new_pred.cpu() != orig_pred.cpu()).flatten()
            # check whether there was enough segments to begin with
            indices = np.array([i.item() for i in _indices if i.item() in enough_segs])
            if len(indices) != len(_indices):
                print(_indices, indices)
                print('found an item with too few segments!')

            if len(indices) > 0:
                # get explainer and explanation
                explainer, explainer_args = get_explainer(explainer_type, mask[indices], device, net)
                # check how much segments are correctly detected
                cor = check_new_explanation(net, explainer, new_adv[indices], new_pred[indices], explainer_args,
                                            mask[indices], aggregation, k, np.array(chosen_segs[indices]).reshape(-1, k))
                for c in cor:
                    res[k][c] += 1
        append_k += str(k)
        nr_of_rem[k] = nr_of_rem_advs

    # show / save results
    print(res, nr_of_rem)
    pickle.dump(nr_of_rem, open(save_path / ('nr_of_remaining_advs_{}.pkl').format(append_k), 'wb'))
    pickle.dump(res, open(save_path / ('label_flips_for_partial_adversaries_{}.pkl').format(append_k), 'wb'))


def prep_paths(opts):
    """ Prepares all necessary paths pointing to data / directories for saving stuff. """
    data_path = Path(opts.data_path)
    adv_path = Path(opts.adv_path) / opts.arch
    results_path = Path(__file__).parent.parent / 'results/experiment6_4'
    perc = opts.subset_percentage
    results_path = results_path / '{}_perc'.format(perc)
    if opts.segment:
        results_path = results_path / '{}_{}_{}_{}_segment'.format(opts.arch, opts.explainer, opts.aggregation, perc)
    else:
        results_path = results_path / '{}_{}_{}_{}'.format(opts.arch, opts.explainer, opts.aggregation, perc)
    if not data_path.exists():
        raise NotADirectoryError('Please define a valid data-path!')
    if not adv_path.exists():
        raise NotADirectoryError('Please define a valid adversarial paths!')
    if not results_path.exists():
        results_path.mkdir(parents=True)
    return adv_path, data_path, results_path


def main():
    # parse arguments
    parser = opts_parser()
    opts = parser.parse_args()
    # get paths
    adv_path, data_path, results_path = prep_paths(opts)
    # start experiment
    run_experiment(data_path, adv_path, opts.explainer, opts.segment, opts.aggregation, opts.ks,
                   opts.arch, results_path, opts.batch_size, opts.subset_percentage)


if __name__ == '__main__':
    main()
