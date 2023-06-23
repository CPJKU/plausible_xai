import torch
import numpy as np

from tqdm import tqdm
from methods.model_utils import get_prediction


def create_new_adversary_with_segments(orig_x, adv_x, mask, top_segments):
    """ Given clean and adversarial file, exchanges clean `top-segments` with adversarial ones. """
    assert len(mask) == len(top_segments)
    # get clean file
    new_adv = orig_x.clone()
    # find top-segments which should be changed to adversarial patches
    indices_to_change = [np.concatenate([np.argwhere(m == s).T for s in ts], axis=0)
                         for m, ts in zip(mask, top_segments)]

    # change clean to adversarial patches
    for i, idx in enumerate(indices_to_change):
        new_adv[i, idx[:, 0], idx[:, 1], idx[:, 2]] = adv_x[i, idx[:, 0], idx[:, 1], idx[:, 2]]

    return new_adv


def get_segment_weights(attrs, mask, aggregation=None):
    """ Returns attributions aggregated for each segment (defined via `mask`). """
    assert len(attrs.shape) == 4, "unexpected shape: {}".format(attrs.shape)
    assert len(mask.shape) == 4, "unexpected shape: {}".format(mask.shape)

    aggregation_dim = [1, 2, 3]
    if aggregation is None:
        aggregation = 'mean'
    available_aggregations = ['mean', 'mean_pos', 'mean_abs', 'magnitude', 'magnitude_mean']
    if aggregation not in available_aggregations:
        raise NotImplementedError("other aggregations than {} are not implemented yet".format(available_aggregations))

    # group attribution-weights according to aggregation method
    segment_weights = []
    for seg in range(len(np.unique(mask))):
        binary_mask = (mask == seg).to(attrs.device)
        intermediate_store = torch.zeros_like(attrs) * torch.nan
        intermediate_store[binary_mask] = attrs[binary_mask]
        assert (~torch.isnan(intermediate_store)).sum() == binary_mask.sum()
        if aggregation == 'mean':
            aggregated_weight = intermediate_store.nanmean(dim=aggregation_dim)
        elif aggregation == 'mean_abs':
            aggregated_weight = intermediate_store.abs().nanmean(dim=aggregation_dim)
        elif aggregation == 'mean_pos':
            positive_mask = (attrs > 0).to(binary_mask.device)
            intermediate_store = torch.zeros_like(attrs) * torch.nan
            intermediate_store[binary_mask & positive_mask] = attrs[binary_mask & positive_mask]
            aggregated_weight = intermediate_store.nanmean(dim=aggregation_dim)
        elif aggregation == 'magnitude':
            aggregated_weight = (intermediate_store**2).nansum(dim=aggregation_dim)
        elif aggregation == 'magnitude_mean':
            aggregated_weight = (intermediate_store**2).nanmean(dim=aggregation_dim)
        segment_weights.append(aggregated_weight.detach().cpu().numpy())
    segment_weights = np.array(segment_weights).T
    segment_weights = np.nan_to_num(segment_weights, nan=-100000)
    return segment_weights


def get_attributions_for_segment_weights(mask, segment_weights):
    """ Gets all attributions for certain segments. """
    assert mask.shape[0] == segment_weights.shape[0] and len(np.unique(mask)) == segment_weights.shape[1]
    attrs = torch.zeros_like(mask, dtype=torch.float32)
    for seg in range(len(np.unique(mask))):
        binary_mask = (mask == seg)
        attrs[binary_mask] = segment_weights[:, seg]
    return attrs


def prepare_k(k_segments, mask):
    """ Adapts k based on whether we have absolute values or percentages. """
    k = k_segments
    if isinstance(k_segments, float):
        k = [int(len(np.unique(m)) * k_segments) for m in mask]
    if isinstance(k_segments, int):
        k = [k for _ in mask]
    return k


def get_top_segments(segment_weights, k):
    """ Returns k highest segment-indices based on segment-weights. """
    assert not np.isnan(segment_weights).any(), 'segment_weights cannot have nan values'
    return [seg[-kk:] for kk, seg in zip(k, np.argsort(segment_weights))]


def adding_k_segments(net, k_segments, adv_loader, device, attrs_dict=None, agg_method=None, use_mag=False):
    """ Adds k adversarial segments to clean files. """
    if use_mag:
        assert agg_method in ['magnitude', 'magnitude_mean']

    positive_k = []
    collect_segment_weights = []

    label_flips = 0
    for i, (orig_x, adv_x, mask) in tqdm(enumerate(adv_loader)):
        orig_x, adv_x = orig_x.to(device), adv_x.to(device)
        orig_pred, _ = get_prediction(net, orig_x)
        top_segments = -1
        k = prepare_k(k_segments, mask)
        if k == 'positive' or not use_mag:
            attrs = attrs_dict[i].to(device)
            segment_weights = get_segment_weights(attrs, mask, aggregation=agg_method)

            if k == 'positive':
                k = (segment_weights > 0).sum(axis=1)
                # print('selected "positive", setting k to {}'.format(k))
                positive_k.append(k)
                collect_segment_weights.append(segment_weights)
            if not use_mag:
                top_segments = get_top_segments(segment_weights, k)

        if use_mag:
            # get 'k' highest segments (according to magnitude)
            delta = torch.abs(orig_x - adv_x).detach().cpu().squeeze()
            segment_mags = get_segment_weights(delta, mask, aggregation=agg_method)
            top_segments = get_top_segments(segment_mags, k)

        assert isinstance(top_segments, list)

        # create new adversary
        new_adv = create_new_adversary_with_segments(orig_x, adv_x, mask, top_segments)
        # get new prediction of network
        new_pred, _ = get_prediction(net, new_adv)

        label_flips += (orig_pred != new_pred).sum()

    label_flips_percentage = (label_flips / len(adv_loader.dataset)).item()
    if k_segments == 'positive':
        return label_flips_percentage, positive_k, collect_segment_weights
    return label_flips_percentage
