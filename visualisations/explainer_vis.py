import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

from methods.results_utils import explainer_map
from visualisations.aggregation_vis import show_attr
from methods.model_utils import get_net, get_prediction
from methods.image_utils import get_tensor_from_filename
from methods.explainer_utils import get_explainer, get_mask
from methods.attribution import attribute_image_features


def compute_explanations(device, explainers, hector, mask, mask_slic, model, pred):
    """ Computes explanations for list of explainers and given exemplary image. """
    attrs_dict = {}
    for i, expl in enumerate(explainers):
        if expl == 'lime_seg':
            explainer, explainer_args = get_explainer('lime', mask_slic, device, model)
        else:
            explainer, explainer_args = get_explainer(expl, mask, device, model)
        attrs = attribute_image_features(model, explainer, hector, pred[0], **explainer_args)
        attrs_dict[expl] = attrs
    return attrs_dict


def plot_figure1(attrs_dict, crop, hector, hector_img, title_name_mapping):
    """ Function that constructs Figure 1. """
    default_cmap = LinearSegmentedColormap.from_list(
        "RdWhGn", ["red", "white", "green"]
    )
    explainers = ['lime', 'lime_seg', 'ig', 'saliency', 'gbp']
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
    fig.tight_layout()
    axes = axes.ravel()
    fontsize = 10
    # visualize original input image
    axes[0].imshow(crop(hector_img))
    axes[0].set_title('Input Image', fontsize=fontsize)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    # visualize attributions
    for i, expl in enumerate(explainers):
        attrs = attrs_dict[expl]
        print(attrs.min(), attrs.max())
        fig, _ = show_attr(attrs.detach().cpu(), hector.detach().cpu(), plt_fig_axis=(fig, axes[i + 1]),
                           use_pyplot=False, alpha_overlay=0.8)
        axes[i + 1].set_title(title_name_mapping[expl], fontsize=fontsize)
    fig.subplots_adjust(right=0.8, wspace=0.1, top=0.83, hspace=0.29)
    cbar_ax = fig.add_axes([0.85, 0.05, 0.02, 0.8])
    dummy_image = np.ones((224, 224, 1))
    dummy_image[:, 1, 0] = -1
    im = plt.imshow(dummy_image, cmap=default_cmap, aspect='auto')
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=6)
    plt.show()


def plot_figurea1(attrs_dict, crop, hector, hector_img, title_name_mapping):
    """ Function that constructs Figure A1 (Appendix). """
    default_cmap = LinearSegmentedColormap.from_list(
        "RdWhGn", ["red", "white", "green"]
    )
    explainers = ['saliency_sm', 'saliency_smsq', 'saliency_var', 'ig_sm', 'ig_smsq',
                  'ig_var', 'gbp_sm', 'gbp_smsq', 'gbp_var']
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), dpi=300)
    fig.tight_layout()
    axes = axes.ravel()
    fontsize = 4
    axes[0].axis('off')
    axes[2].axis('off')
    # visualize original input image
    axes[1].imshow(crop(hector_img))
    axes[1].set_title('Input Image', fontsize=fontsize)
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    # visualize attributions
    for i, expl in enumerate(explainers):
        attrs = attrs_dict[expl]
        fig, _ = show_attr(attrs.detach().cpu(), hector.detach().cpu(), plt_fig_axis=(fig, axes[i + 3]),
                           use_pyplot=False, alpha_overlay=0.8)
        axes[i + 3].set_title(title_name_mapping[expl], fontsize=fontsize)
    fig.subplots_adjust(right=0.8, wspace=0., top=0.9, hspace=0.3)
    cbar_ax = fig.add_axes([0.85, 0.05, 0.02, 0.8])
    dummy_image = np.ones((224, 224, 1))
    dummy_image[:, 1, 0] = -1
    im = plt.imshow(dummy_image, cmap=default_cmap, aspect='auto')
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=5)
    plt.show()


def main():
    # setting some constants (model type, exemplary image...)
    device = torch.device('cpu')
    model_name = 'vgg'
    model = get_net(model_name, device)
    hector = get_tensor_from_filename(Path(__file__).parent.parent / 'figs/hector.jpg',
                                      inception=(model_name == 'inceptionv3')).unsqueeze(0).to(device)
    hector_img = Image.open(Path(__file__).parent.parent / 'figs/hector.jpg').convert("RGB")
    pred = get_prediction(model, hector)

    # prepare explainers, compute explanations
    mask = get_mask(image=hector, image_size=hector.shape[3])
    mask_slic = get_mask(rectangular=False, image=hector.detach().cpu(), image_size=hector.shape[3])
    explainers = ['lime', 'lime_seg', 'ig', 'saliency', 'gbp', 'saliency_sm', 'saliency_smsq', 'saliency_var',
                  'ig_sm', 'ig_smsq', 'ig_var',
                  'gbp_sm', 'gbp_smsq', 'gbp_var']

    attrs_dict = compute_explanations(device, explainers, hector, mask, mask_slic, model, pred)

    # prepare plotting
    crop = transforms.Compose([
        transforms.Resize(299 if model_name == 'inceptionv3' else 256),
        transforms.CenterCrop(hector.shape[3])  # the tensor is already correctly cropped, the PIL image isn't
    ])
    title_name_mapping = {}
    for key, value in explainer_map.items():
        title_name_mapping[key.replace('_mean_abs', '').replace('_mean', '')] = value

    title_name_mapping['lime_seg'] = 'LIME (SLIC)'

    # do actual plotting
    plot_figure1(attrs_dict, crop, hector, hector_img, title_name_mapping)
    plot_figurea1(attrs_dict, crop, hector, hector_img, title_name_mapping)


if __name__ == '__main__':
    main()
