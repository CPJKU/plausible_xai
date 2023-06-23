import torch

from tqdm import tqdm
from pathlib import Path
from torchvision import datasets
from argparse import ArgumentParser
from methods.model_utils import get_net
from torch.utils.data import DataLoader
from methods.image_utils import transform_inceptionv3, transform_imagenet


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Script that computes clean accuracy for (all) architectures.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data-path', metavar='DIR', type=str, help='Path to ImageNet data.', required=True)
    parser.add_argument('--arch', nargs='+', default=['alexnet', 'resnet', 'vgg', 'densenet', 'inceptionv3'],
                        help='Which architectures to look at - define one or multiple of [alexnet, vgg, resnet, densenet, inceptionv3]')
    return parser


def process_model(model, valid_loader, device):
    """ Computes accuracy for a particular model (architecture). """
    num_correct = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(valid_loader)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.data.max(1)[1]
            num_correct += pred.eq(y.data).sum().item()
    return num_correct


def main():
    # parse arguments
    parser = opts_parser()
    options = parser.parse_args()

    data_path = Path(options.data_path)
    archs = options.arch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for arch in archs:
        net = get_net(arch, device)
        if arch == 'inceptionv3':
            valid_loader = DataLoader(datasets.ImageNet(data_path, 'val', transform=transform_inceptionv3),
                                      batch_size=128, shuffle=True)
        else:
            valid_loader = DataLoader(datasets.ImageNet(data_path, 'val', transform=transform_imagenet),
                                      batch_size=128, shuffle=True)
        num_samples = len(valid_loader.dataset)

        num_correct = process_model(net, valid_loader, device)
        print("model_name: {}; correct: {}; len: {}; accuracy: {:.2f}%".format(arch, num_correct, num_samples,
                                                                               num_correct / num_samples * 100))


if __name__ == '__main__':
    main()
