import torch
import numpy as np

from copy import deepcopy
from skimage.segmentation import slic
from captum._utils.models import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum.attr import Lime, IntegratedGradients, Saliency, GuidedBackprop, NoiseTunnel


def get_lime_explainer_params(net, mask, device, similarity_function=None):
    """LIME has a bit more parameters, therefore it's a separate function"""
    lime_args = {
        'n_samples': 512,
        'feature_mask': mask.to(device),
        'perturbations_per_eval': 16,
        'show_progress': False
    }
    # print('similarity_function', similarity_function)
    exp_eucl_distance = similarity_function
    if exp_eucl_distance is None:
        exp_eucl_distance = get_exp_kernel_similarity_function('cosine', kernel_width=0.25)
    lr_lime = Lime(net, interpretable_model=SkLearnLinearRegression(), similarity_func=exp_eucl_distance)
    return lr_lime, lime_args


def get_explainer(explainer_name, mask, device, net, lime_similarity_function=None):
    """ Returns an explainer and describing arguments. """

    if explainer_name != 'lime' and lime_similarity_function is not None:
        raise ValueError('lime_similarity_function can only be set for lime')

    def get_nt_params(nt_type):
        nt_params = {'nt_type': nt_type, 'nt_samples': 25}
        return nt_params

    def update_inplace(args, add_args):
        copied_args = deepcopy(args)
        copied_args.update(add_args)
        return copied_args

    ig_args = {
        'baselines': 0.0,
        'internal_batch_size': 10
    }

    saliency_args = {
        'abs': False
    }

    explainers = {
        'lime': (get_lime_explainer_params(net, mask, device, lime_similarity_function)),
        'saliency': (Saliency(net), saliency_args),
        'ig': (IntegratedGradients(net), ig_args),
        'gbp': (GuidedBackprop(net), {}),
        'saliency_sm': (NoiseTunnel(Saliency(net)), update_inplace(saliency_args, get_nt_params('smoothgrad'))),
        'saliency_smsq': (NoiseTunnel(Saliency(net)), update_inplace(saliency_args, get_nt_params('smoothgrad_sq'))),
        'saliency_var': (NoiseTunnel(Saliency(net)), update_inplace(saliency_args, get_nt_params('vargrad'))),
        'ig_sm': (NoiseTunnel(IntegratedGradients(net)), update_inplace(ig_args, get_nt_params('smoothgrad'))),
        'ig_smsq': (NoiseTunnel(IntegratedGradients(net)), update_inplace(ig_args, get_nt_params('smoothgrad_sq'))),
        'ig_var': (NoiseTunnel(IntegratedGradients(net)), update_inplace(ig_args, get_nt_params('vargrad'))),
        'gbp_sm': (NoiseTunnel(GuidedBackprop(net)), get_nt_params('smoothgrad')),
        'gbp_smsq': (NoiseTunnel(GuidedBackprop(net)), get_nt_params('smoothgrad_sq')),
        'gbp_var': (NoiseTunnel(GuidedBackprop(net)), get_nt_params('vargrad'))
    }

    return explainers[explainer_name]


def get_mask(rectangular=True, image=None, image_size=224):
    """ Returns a mask describing different segments of an image; either segmentation or rectangular segments. """
    if rectangular:
        segs_per_dim = 4
        mod = image_size // segs_per_dim

        mask = np.zeros((image_size, image_size))
        for i in range(image_size):
            for j in range(image_size):
                mask[i, j] = (i // mod * segs_per_dim + j // mod)
                if i >= mod * segs_per_dim:
                    mask[i, j] -= segs_per_dim
                if j >= mod * segs_per_dim:
                    mask[i, j] -= 1
        mask = np.repeat(np.repeat(mask[None, :, :], 3, axis=0)[None, :, :, :], len(image), axis=0)
        mask = torch.tensor(mask, dtype=int)
    else:
        image = image.numpy()
        segments = slic(image, n_segments=16, compactness=10., sigma=1., channel_axis=1)
        segments = np.repeat(segments[:, None, :, :], 3, axis=1) - 1
        mask = torch.tensor(segments, dtype=int)
    return mask
