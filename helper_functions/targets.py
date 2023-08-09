import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path
from helper_functions import ownutilities, frame_utils


def zero_flow(flow):
    """Create a zero tensor with the same size as flow

    Args:
        flow (tensor): input

    Returns:
        tensor: containing zeros with same dimension as the input
    """
    return torch.zeros_like(flow)

def get_target(target_name, flow_pred_init, device=None):
    """Getter method which yields a specified target flow used during PCFA

    Args:
        target_name (str):
            description the attack target. Options: [zero | negative | custom]
        flow_pred_init (tensor):
            unattacked flow field
        custom_target_path (str, optional):
            if custom target is desired provide the path to a .npy perturbation file. Defaults to "".
        device (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: Undefined choice for target.

    Returns:
        tensor: target flow field used during PCFA
    """
    if target_name == 'zero':
        target = zero_flow(flow_pred_init)
    else:
        raise ValueError('The specified target type "' + target_name + '" is not defined and cannot be used. Select "zero". Aborting.')
    return target
