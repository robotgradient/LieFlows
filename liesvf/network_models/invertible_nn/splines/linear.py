from liesvf.utils import math as math

import torch
from torch.nn import functional as F


from .base import InputOutsideDomain


def linear_splines(inputs, unnormalized_pdf,
                  left=0., right=1., bottom=0., top=1.):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    if (torch.min(inputs) < left or torch.max(inputs) > right):
        print('Input Outside Domain. Return')
        return inputs
        #raise InputOutsideDomain()


    inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.
    cdf = F.pad(cdf, pad=(1, 0), mode='constant', value=0.0)


    ## BINS ##
    bin_pos = inputs * num_bins

    bin_idx = torch.floor(bin_pos).long()
    bin_idx[bin_idx >= num_bins] = num_bins - 1

    alpha = bin_pos - bin_idx.float()

    input_pdfs = pdf.gather(-1, bin_idx[..., None])[..., 0]

    outputs = cdf.gather(-1, bin_idx[..., None])[..., 0]
    outputs += alpha * input_pdfs
    outputs = torch.clamp(outputs, 0, 1)

    ###########

    outputs = outputs * (top - bottom) + bottom

    return outputs


def linear_spline(inputs, unnormalized_pdf,
                  inverse=False,
                  left=0., right=1., bottom=0., top=1.):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """

    if (torch.min(inputs) < left or torch.max(inputs) > right):
        print('Input Outside Domain. Return')
        return inputs

    # if not inverse and min_bool or max_bool:
    #     print('Input Outside Domain. Return')
    #     return inputs
    # elif inverse and min_bool or max_bool:
    #     print('Input Outside Domain. Return')
    #     return inputs

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.
    cdf = F.pad(cdf, pad=(1, 0), mode='constant', value=0.0)

    if inverse:
        inv_bin_idx = math.searchsorted(cdf, inputs)

        bin_boundaries = (torch.linspace(0, 1, num_bins+1)
                          .view([1] * inputs.dim() + [-1])
                          .expand(*inputs.shape, -1))

        slopes = ((cdf[..., 1:] - cdf[..., :-1])
                  / (bin_boundaries[..., 1:] - bin_boundaries[..., :-1]))
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx.unsqueeze(-1)
        input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
        input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = torch.clamp(outputs, 0, 1)
    else:
        bin_pos = inputs * num_bins

        bin_idx = torch.floor(bin_pos).long()
        bin_idx[bin_idx >= num_bins] = num_bins - 1

        alpha = bin_pos - bin_idx.float()

        input_pdfs = pdf.gather(-1, bin_idx[..., None])[..., 0]

        outputs = cdf.gather(-1, bin_idx[..., None])[..., 0]
        outputs += alpha * input_pdfs
        outputs = torch.clamp(outputs, 0, 1)

    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs