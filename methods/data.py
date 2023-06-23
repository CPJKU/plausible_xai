import torch
import warnings

import numpy as np
import torchvision.transforms as transforms

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from methods.subset import AdversarialImagenetSubset
from methods.image_utils import get_tensor_from_filename

N_FULL_SET = 50000


class AdversarialImagenet(Dataset):
    """ Dataset for adversarial data. """
    def __init__(self, root_path, in_files, inception=False, segment=False):
        file_list = [(Path(in_files[i][0]),
                      Path(root_path) / Path(in_files[i][0]).parent.name / Path(in_files[i][0]).name)
                     for i in range(len(in_files))]
        self.file_list = [(f, a.with_suffix('.npy')) for f, a in file_list if a.with_suffix('.npy').exists()]
        self.inception = inception
        self.segment = segment

    def __len__(self):
        return len(self.file_list)

    def get_mask(self, adv_file):
        """ Gets rectangular or segmentation mask for particular adversary. """
        image_size = 224 if not self.inception else 299

        if not self.segment:
            # get mask with rectangles
            segs_per_dim = 4
            mod = image_size // segs_per_dim

            mask = np.zeros((image_size, image_size))
            for i in range(image_size):
                for j in range(image_size):
                    mask[i, j] = (i // mod * segs_per_dim + j // mod)

                    # fix border for masks that can't be divided by segs_per_dim
                    if i >= mod * segs_per_dim:
                        mask[i, j] -= segs_per_dim
                    if j >= mod * segs_per_dim:
                        mask[i, j] -= 1
            mask = np.repeat(mask[None, :, :], 3, axis=0)
            return torch.tensor(mask, dtype=int)

        else:
            # load segmentation and return mask
            mask_file = str(adv_file).replace('adversaries', 'masks')
            if not Path(mask_file).exists():
                warnings.warn('Could not find segmentation file, returns 0-mask!')
                mask = np.zeros((3, image_size, image_size))
            else:
                mask = np.repeat(np.load(mask_file)[None, :, :], 3, axis=0)
            return torch.tensor(mask, dtype=int)

    def __getitem__(self, index):
        orig_file, adv_file = self.file_list[index]
        # load original file
        orig = get_tensor_from_filename(str(orig_file), self.inception)
        # load adversary
        adv = np.load(str(adv_file)).transpose(1, 2, 0)
        adv = adversarial_transform(adv)
        # get mask
        mask = self.get_mask(adv_file)
        return orig, adv, mask


def adversarial_transform(img):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )(img)


def get_adv_imagenet_loader(root_path, imagenet_files, batch_size=1, shuffle=False, subset_percentage=100,
                            inception=False, segment=False):
    """ Returns dataloader that loads adversaries (in same order as given imagenet dataset, if not shuffled). """
    assert subset_percentage > 0 and subset_percentage <= 100, \
        "subset_percentage ({}) out of range (0, 100]".format(subset_percentage)
    dataset = AdversarialImagenet(root_path, imagenet_files, inception=inception, segment=segment)
    if subset_percentage < 100:
        n_samples = len(dataset)
        take_n_samples = int(N_FULL_SET * subset_percentage / 100)  # take x% of the full validation set
        print('taking {} ({} %) samples'.format(take_n_samples, subset_percentage))
        random_indices = np.random.choice(np.arange(n_samples), take_n_samples, replace=False)
        dataset = AdversarialImagenetSubset(dataset, indices=random_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
