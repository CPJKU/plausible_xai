import torch
import pickle
import numpy as np
import torch.nn as nn

from pathlib import Path
from torchvision import datasets
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from methods.model_utils import get_net, get_prediction
from methods.image_utils import transform_imagenet as transform

from captum.robust import PGD


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Search for hyper-parameters for adversarial attacks on ImageNet.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data-path', metavar='DIR', type=str, help='Path to ImageNet data.', required=True)
    parser.add_argument('--arch', type=str, default='alexnet',
                        help='Which network to use. One of [alexnet, vgg, resnet, densenet, inceptionv3].')
    return parser


def do_grid_search(device, net, nr_samples_to_check, radi, step_nrs, step_sizes, valid_loader):
    """ Performs grid search for hyper-parameters of adversarial attack. """
    res = {}
    for step_size in step_sizes:
        for radius in radi:
            for step_nr in step_nrs:
                diffs = []
                successful = 0
                print('cur: {}, {}, {}'.format(step_size, radius, step_nr))
                for i, (x, y) in enumerate(valid_loader):
                    x, y = x.to(device), y.to(device)
                    bef_pred, _ = get_prediction(net, x)
                    target = torch.tensor(np.random.choice([j for j in range(1000) if j != bef_pred], 1)).to(device)
                    lower = x.min()
                    upper = x.max()

                    pgd = PGD(net, nn.CrossEntropyLoss(reduction='none'), lower_bound=lower, upper_bound=upper)
                    adv = pgd.perturb(inputs=x, radius=radius, step_size=step_size, step_num=step_nr, target=target,
                                      targeted=True)
                    aft_pred, _ = get_prediction(net, adv)

                    if aft_pred.item() == target.item():
                        # count only successful attacks
                        successful += 1
                        diffs.append(np.sum((np.abs(x.cpu() - adv.cpu()) ** 2).numpy()))

                    if i == nr_samples_to_check:
                        break
                res.update({'{}_{}_{}'.format(step_size, radius, step_nr): [successful, np.mean(diffs)]})
                print('successful: {}, mean differences: {}'.format(successful, np.mean(diffs)))
    return res


def main():
    torch.manual_seed(12)
    # parse arguments
    parser = opts_parser()
    options = parser.parse_args()

    # preps
    data_path = Path(options.data_path)
    arch = options.arch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_net(arch, device)
    valid_loader = DataLoader(datasets.ImageNet(data_path, 'val', transform=transform), batch_size=1, shuffle=True)

    # subset of ImageNet data that we want to use for hyper-parameter search
    nr_samples_to_check = 1000
    # define hyper-parameters we want to search over
    step_sizes = [0.02, 0.01, 0.025]
    radi = [0.13, 0.1, 0.05, 0.01]
    step_nrs = [100]

    # store results
    res = do_grid_search(device, net, nr_samples_to_check, radi, step_nrs, step_sizes, valid_loader)
    pickle.dump(res, open('hyperparams_{}.pkl'.format(arch), 'wb'))


if __name__ == '__main__':
    main()
