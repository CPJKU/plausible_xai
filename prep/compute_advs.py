import csv
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from captum.robust import PGD
from argparse import ArgumentParser

from torchvision import datasets
from torch.utils.data import DataLoader

from methods.model_utils import get_net, get_prediction
from methods.image_utils import transform_imagenet as transform, transform_inceptionv3, unnormalise


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Runs adversarial (PGD) attack on image-net data (for precomputed model).'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data-path', metavar='DIR', type=str, help='Path to image-net data.', required=True)
    parser.add_argument('--save-path', metavar='DIR', type=str, default=Path(__file__).parent.parent / 'adversaries/',
                        help='Path to directory where adversaries should be stored.')
    parser.add_argument('--arch', type=str, default='alexnet',
                        help='Which network to use. One of [alexnet, vgg, resnet, densenet, inceptionv3].')
    parser.add_argument('--radius', type=float, default=0.10, help='Radius of attack.')
    parser.add_argument('--stepsize', type=float, default=0.01, help='Step-size of attack.')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to take during attack.')
    parser.add_argument('--fix-lb', type=float, required=False, help='Fixed lower bound for attack.')
    parser.add_argument('--fix-ub', type=float, required=False, help='Fixed upper bound for attack.')
    parser.add_argument('--nrsamples', type=int, default=50000, help='Number of samples we compute adversary for.')

    return parser


def attack(data_path, save_path, radius, step_size, step_nr, lower, upper, nr_samples, arch):
    """ Performs PGD attack; stores log file with new/old labels, and adversarial images. """
    np.random.seed(21)
    log = [['file name', 'label', 'before', 'target', 'after']]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = get_net(arch, device)
    if arch == 'inceptionv3':
        valid_loader = DataLoader(datasets.ImageNet(data_path, 'val', transform=transform_inceptionv3),
                                  batch_size=1, shuffle=False)
    else:
        valid_loader = DataLoader(datasets.ImageNet(data_path, 'val', transform=transform), batch_size=1, shuffle=False)

    # run through samples
    with tqdm(total=nr_samples) as pbar:
        for i, (x, y) in enumerate(valid_loader):
            pbar.update(1)
            x, y = x.to(device), y.to(device)
            bef_pred, _ = get_prediction(net, x)
            target = torch.tensor(np.random.choice([j for j in range(1000) if j != bef_pred], 1)).to(device)
            print('y: {}, target: {}'.format(y.item(), target.item()))
            # if not fixed, get current min / max values as bounds for attack
            if not lower:
                lower = x.min()
            if not upper:
                upper = x.max()
            pgd = PGD(net, nn.CrossEntropyLoss(reduction='none'), lower_bound=lower, upper_bound=upper)
            adv = pgd.perturb(inputs=x, radius=radius, step_size=step_size, step_num=step_nr, target=target,
                              targeted=True)
            aft_pred, _ = get_prediction(net, adv)

            # save adversary
            orig_name = Path(valid_loader.dataset.samples[i][0])
            if not (save_path / orig_name.parent.name).exists():
                (save_path / orig_name.parent.name).mkdir()
            # only save if target was reached
            if aft_pred.item() == target.item():
                np.save(save_path / orig_name.parent.name / orig_name.with_suffix('.npy').name, unnormalise(adv).squeeze().cpu())
            # save infos
            log.append([orig_name.parent.name + '/' + orig_name.name, y.item(), bef_pred.item(),
                        target.item(), aft_pred.item()])

            if i == nr_samples - 1:
                break

    # save log file
    with open(save_path / 'log.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(log)


def main():
    # parse arguments
    parser = opts_parser()
    options = parser.parse_args()

    # prepare paths
    data_path = Path(options.data_path)
    save_path = Path(options.save_path) / '{}'.format(options.arch)

    if not data_path.exists():
        raise NotADirectoryError('Please define a valid data-path!')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # run attack
    attack(data_path, save_path, options.radius, options.stepsize, options.steps, options.fix_lb,
           options.fix_ub, options.nrsamples, options.arch)


if __name__ == '__main__':
    main()
