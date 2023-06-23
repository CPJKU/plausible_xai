import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def transform_imagenet(img):
    """ Method to normalize an image to Imagenet mean and standard deviation. """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            ),
        ]
    )(img)


def transform_inceptionv3(img):
    """ Method to normalize an image to Imagenet mean and standard deviation for Inceptionv3 net. """
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    )(img)


def unnormalise(x):
    """ Performs opposite of normalisation. """
    device = x.device
    x = x.squeeze() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device) + \
        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    return x.clamp(0, 1)


def get_tensor_from_filename(filename, inception=False):
    """ Loads image from filename and transforms (incl. normalises) it.  """
    img = Image.open(filename).convert("RGB")
    if not inception:
        return transform_imagenet(img)
    else:
        return transform_inceptionv3(img)


def image_show(img, pred):
    """ Plots a given image (and its prediction). """
    normalize01 = lambda x: (x - x.min()) / (x.max() - x.min())
    npimg = img.squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(normalize01(npimg))
    plt.title("prediction: %s" % pred)
    plt.show()


def adversarial_image_show(orig, adv, orig_pred, adv_pred, normalize=None):
    """ Plots original / adversarial image (and their difference). """
    if normalize is None:
        normalize = lambda x: x
    fig, axs = plt.subplots(1, 3, figsize=(20, 60))
    axs[0].set_title('original (pred: {})'.format(orig_pred))
    orig_img = orig.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    axs[0].imshow(normalize(orig_img))
    axs[1].set_title('adversary (pred: {})'.format(adv_pred))
    adv_img = adv.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    axs[1].imshow(normalize(adv_img))
    axs[2].set_title('diff')
    axs[2].imshow(normalize(np.abs(orig_img - adv_img)))
    plt.show()
