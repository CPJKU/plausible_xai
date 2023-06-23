import torch

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from methods.model_utils import get_net, get_prediction
from methods.attribution import attribute_image_features
from methods.image_utils import get_tensor_from_filename
from methods.explainer_utils import get_explainer, get_mask
from methods.segments import get_segment_weights, get_attributions_for_segment_weights


def show_attr(attr_map, orig, method='blended_heat_map', **kwargs):
    """ Visualises a particular attribution map. """
    attr_map = attr_map.squeeze()
    orig = orig.squeeze()
    return viz.visualize_image_attr(
       attr_map.permute(1, 2, 0).numpy(),  # adjust shape to height, width, channels
       original_image=orig.squeeze().permute(1, 2, 0).numpy(),
       method=method, sign='all', **kwargs)


def display_axis(ax, img, title):
    """ Method for displaying original image. """
    ax.imshow(img)
    ax.set_title(title, fontsize=6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


def display_explainer(helix, attrs, title_name_mapping, fig, ax, expl):
    """ Wrapper method to visualise attribution map of an explainer on top of image, and sets title appropriately. """
    fig, _ = show_attr(attrs[expl].detach(), helix.detach(), alpha_overlay=0.8,
                       plt_fig_axis=(fig, ax), use_pyplot=False)
    ax.set_title(title_name_mapping[expl], fontsize=6)
    return fig


def compute_agg_attrs(attrs, available_aggregations, mask, selected_explainer, slic_mask, n_pixels):
    """ Given standard feature attributions, computes all possible aggregations thereof (for slic and rectangular segments). """
    attrs_dict = {}
    for current_aggregation in available_aggregations:
        # rectangles
        segment_weights = torch.tensor(get_segment_weights(attrs, mask, current_aggregation))
        agg_attrs = get_attributions_for_segment_weights(mask, segment_weights)
        attrs_dict['{}_{}'.format(selected_explainer, current_aggregation)] = agg_attrs

        # slic
        segment_weights = torch.tensor(get_segment_weights(attrs, slic_mask, current_aggregation))
        agg_attrs = get_attributions_for_segment_weights(slic_mask, segment_weights)
        attrs_dict['{}_{}_slic'.format(selected_explainer, current_aggregation)] = agg_attrs

    for current_aggregation in available_aggregations:
        attrs_dict['{}_{}_top3'.format(selected_explainer, current_aggregation)] = keep_topk_3channel(
            attrs_dict['{}_{}'.format(selected_explainer, current_aggregation)], n_pixels)
        attrs_dict['{}_{}_slic_top3'.format(selected_explainer, current_aggregation)] = keep_topk_3channel(
            attrs_dict['{}_{}_slic'.format(selected_explainer, current_aggregation)], n_pixels)

    return attrs_dict


def keep_topk_1channel(flattened_attr, n_pixels, replace_by=None):
    """ Helper method to recreate aggregation of Göpfert et al. """
    topk_indices = torch.topk(flattened_attr, n_pixels)[1]

    pixel_attr = torch.zeros_like(flattened_attr)

    if replace_by is None:
        pixel_attr[topk_indices] = flattened_attr[topk_indices]
    else:
        pixel_attr[topk_indices] = replace_by

    pixel_attr = pixel_attr.reshape((1, 1, 224, 224))
    pixel_attr = pixel_attr.repeat((1, 3, 1, 1))
    return pixel_attr


def keep_topk_3channel(attr, n_pixels, replace_by=None):
    """ Helper method to get top-3 aggregated attribution segments. """
    flattened_attr = attr.detach().cpu().reshape(3 * 224**2)

    topk_indices = torch.topk(flattened_attr, 3 * n_pixels)[1]

    pixel_attr = torch.zeros_like(flattened_attr)

    if replace_by is None:
        pixel_attr[topk_indices] = flattened_attr[topk_indices]
    else:
        pixel_attr[topk_indices] = replace_by

    pixel_attr = pixel_attr.reshape((1, 3, 224, 224))
    return pixel_attr


def plot_aggregations(available_aggregations, agg_attrs, helix, helix_img, model_name, selected_explainer):
    """ Function that constructs complete Figure 4. """
    crop = transforms.Compose([
        transforms.Resize(299 if model_name == 'inceptionv3' else 256),
        transforms.CenterCrop(helix.shape[3])  # the tensor is already correctly cropped, the PIL image isn't
    ])
    title_name_mapping = {
        'ig': 'IG',
        'ig_pixel': 'Pixelised Attributions (IG)',
        'ig_mean_top3': 'Top3 (IG, rect., mean)',
        'ig_mean_abs_top3': 'Top3 (IG, rect., mean_abs)',
        'ig_mean_pos_top3': 'Top3 (IG, rect., mean_pos)',
        'ig_mean_slic_top3': 'Top3 (IG, SLIC, mean)',
        'ig_mean_abs_slic_top3': 'Top3 (IG, SLIC, mean_abs)',
        'ig_mean_pos_slic_top3': 'Top3 (IG, SLIC, mean_pos)'
    }

    # start plotting
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=300)
    plt.tight_layout()
    axes = axes.ravel()
    # visualize original input image
    display_axis(axes[0], crop(helix_img), 'Input Image')
    # visualize selected explainer
    fig = display_explainer(helix, agg_attrs, title_name_mapping, fig, axes[1], selected_explainer)
    # visualize selected explainer
    fig = display_explainer(helix, agg_attrs, title_name_mapping, fig, axes[2], '{}_pixel'.format(selected_explainer))
    # visualize aggregations of selected explainer
    for i in range(3):
        fig = display_explainer(helix, agg_attrs, title_name_mapping, fig, axes[i + 3],
                                '{}_{}_top3'.format(selected_explainer, available_aggregations[i]))
    for i in range(3):
        fig = display_explainer(helix, agg_attrs, title_name_mapping, fig, axes[i + 6],
                                '{}_{}_slic_top3'.format(selected_explainer, available_aggregations[i]))
    fig.subplots_adjust(right=0.8, hspace=0.3, top=0.9)
    cbar_ax = fig.add_axes([0.85, 0.05, 0.02, 0.8])
    dummy_image = np.ones((224, 224, 1))
    dummy_image[:, 1, 0] = -1
    default_cmap = LinearSegmentedColormap.from_list(
        "RdWhGn", ["red", "white", "green"]
    )
    im = plt.imshow(dummy_image, cmap=default_cmap, aspect='auto')
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=6)
    plt.show()


def main():
    # define which explainer and model we take for this exemplary visualisation
    selected_explainer = 'ig'
    model_name = 'vgg'
    device = torch.device('cpu')
    model = get_net(model_name, device)
    available_aggregations = ['mean', 'mean_pos', 'mean_abs']
    n_pixels = int(224 * 224 / 16 * 3)

    # load image we use
    helix = get_tensor_from_filename(Path(__file__).parent.parent / 'figs/helix.jpg').unsqueeze(0).to(device)
    helix_img = Image.open(Path(__file__).parent.parent / 'figs/helix.jpg').convert("RGB")
    # get prediction
    pred = get_prediction(model, helix)
    # get masks for image
    mask = get_mask(rectangular=True, image=helix, image_size=helix.shape[3])
    slic_mask = get_mask(rectangular=False, image=helix, image_size=helix.shape[3])

    # compute standard attributions
    explainer, explainer_args = get_explainer(selected_explainer, mask, device, model)
    attrs = attribute_image_features(model, explainer, helix, pred[0], **explainer_args)

    # compute aggregated attributions
    agg_attrs = compute_agg_attrs(attrs, available_aggregations, mask, selected_explainer, slic_mask, n_pixels)
    agg_attrs.update({'ig': attrs})     # add standard attributions to result dict
    # compute Göpfert aggregation for comparison
    flattened_attr = agg_attrs[selected_explainer].sum(axis=1).detach().cpu().reshape(224 ** 2)
    agg_attrs['{}_pixel'.format(selected_explainer)] = keep_topk_1channel(flattened_attr, n_pixels, replace_by=1.0)

    plot_aggregations(available_aggregations, agg_attrs, helix, helix_img, model_name, selected_explainer)


if __name__ == '__main__':
    main()
