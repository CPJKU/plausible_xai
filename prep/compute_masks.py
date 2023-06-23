import numpy as np

from tqdm import tqdm
from pathlib import Path
from torchvision import datasets
from argparse import ArgumentParser
from skimage.segmentation import slic
from methods.data import AdversarialImagenet
from torch.utils.data import Subset, DataLoader


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Pre computes segmentation masks for adversaries.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data-path', metavar='DIR', type=str, help='Path to ImageNet data.', required=True)
    parser.add_argument('--adv-path', metavar='DIR', type=str, help='Path to adversarial data.', required=True)
    parser.add_argument('--arch', type=str, default='alexnet', help='Which network to use. One of [alexnet, vgg, '
                                                                    'resnet, densenet, inceptionv3].')
    parser.add_argument('--start', type=int, default=0, help='Index of starting element for dataloader')
    parser.add_argument('--nr_files', type=int, default=None, help='Number of files which should be handled.')
    return parser


def segment(adv_path, data_path, save_path, start_idx, nr_files):
    """ Segment adversarial files. """
    adv_set = AdversarialImagenet(adv_path, datasets.ImageNet(data_path, 'val').samples)

    if not nr_files:
        nr_files = len(adv_set)
        print('Set nr of files to: {}'.format(nr_files))
    last_file = min(len(adv_set), start_idx + nr_files)
    indices = range(start_idx, last_file)

    data_loader = DataLoader(Subset(adv_set, indices=indices), batch_size=1, shuffle=False)

    for i, (orig, adv, _) in tqdm(zip(indices, data_loader)):
        adv = adv.squeeze().numpy()
        segments = slic(adv, n_segments=16, compactness=10., sigma=1., channel_axis=0, start_label=0)
        save_to = save_path / adv_set.file_list[i][-1].parent.name / adv_set.file_list[i][-1].name
        if not save_to.parent.exists():
            save_to.parent.mkdir()
        np.save(save_to, segments)


def main():
    # parse arguments
    parser = opts_parser()
    opts = parser.parse_args()

    adv_path = Path(opts.adv_path) / opts.arch
    data_path = Path(opts.data_path)
    save_path = adv_path.parent.parent / 'masks' / opts.arch

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    segment(adv_path, data_path, save_path, opts.start, opts.nr_files)


if __name__ == '__main__':
    main()
