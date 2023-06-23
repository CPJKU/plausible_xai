import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import datasets
from argparse import ArgumentParser
from methods.model_utils import get_net
from torch.utils.data import Dataset, DataLoader
from methods.image_utils import get_tensor_from_filename, transform_inceptionv3


def opts_parser():
    """ Prepares argument parsing. """
    desc = 'Script that computes adversarial accuracy for (all) architectures.'
    parser = ArgumentParser(description=desc)
    parser.add_argument('--data-path', metavar='DIR', type=str, help='Path to ImageNet data.', required=True)
    parser.add_argument('--adv-path', metavar='DIR', type=str, required=True,
                        help='Path pointing to directory where adversaries are stored.')
    parser.add_argument('--arch', nargs='+', default=['alexnet', 'resnet', 'vgg', 'densenet', 'inceptionv3'],
                        help='Which architectures to look at - define one or multiple of [alexnet, vgg, resnet, densenet, inceptionv3]')
    return parser


class AdvDataset(Dataset):
    """ Dataset for adversarial and clean data. """
    def __init__(self, root_path, in_files, inception=False):
        file_list = [(Path(in_files[i][0]),
                      Path(root_path) / Path(in_files[i][0]).parent.name / Path(in_files[i][0]).name,
                      in_files[i][-1])
                     for i in range(len(in_files))]
        self.file_list = [(a.with_suffix('.npy'), i) if a.with_suffix('.npy').exists() else (f, i) for f, a, i in file_list]
        self.inception = inception

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name, label = self.file_list[index]
        if str(file_name).endswith('.npy'):
            # load adversary
            adv = np.load(str(file_name)).transpose(1, 2, 0)
            file = adversarial_transform(adv)
        else:
            # load original file
            if not self.inception:
                file = get_tensor_from_filename(str(file_name))
            else:
                img = Image.open(str(file_name)).convert("RGB")
                file = transform_inceptionv3(img)

        return file, label


def adversarial_transform(img):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)


def main():
    # parse arguments
    parser = opts_parser()
    options = parser.parse_args()

    # prepare paths
    data_path = Path(options.data_path)
    adv_base_path = Path(options.adv_path)

    archs = options.arch
    batch_size = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for arch in archs:
        adv_path = adv_base_path / arch
        net = get_net(arch, device)
        data = DataLoader(AdvDataset(adv_path, datasets.ImageNet(data_path, 'val').samples, arch == 'inceptionv3'),
                          batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            num_correct = 0
            for x, y in tqdm(data):
                x, y = x.to(device), y.to(device)
                output = net(x)
                pred = output.data.max(1)[1]
                num_correct += pred.eq(y.data).sum().item()
            print("model_name: {}; correct: {}; len: {}; accuracy: {:.2f}%".format(arch, num_correct, len(data.dataset),
                                                                                   num_correct / len(data.dataset) * 100))


if __name__ == '__main__':
    main()
