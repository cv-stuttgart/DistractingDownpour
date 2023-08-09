import torch

def avg_epe(flow1, flow2):
    """"
    Compute the average endpoint errors (AEE) between two flow fields.
    The epe measures the euclidean- / 2-norm of the difference of two optical flow vectors
    (u0, v0) and (u1, v1) and is defined as sqrt((u0 - u1)^2 + (v0 - v1)^2).

    Args:
        flow1 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component
        flow2 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component

    Raises:
        ValueError: dimensons not valid

    Returns:
        float: scalar average endpoint error
    """
    diff_squared = (flow1 - flow2)**2
    if len(diff_squared.size()) == 3:
        # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.mean(torch.sum(diff_squared, dim=0).sqrt())
    elif len(diff_squared.size()) == 4:
        # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.mean(torch.sum(diff_squared, dim=1).sqrt())
    else:
        raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(flow1.size()) + " and " + str(flow1.size()))
    return epe

def avg_mse(flow1, flow2):
    """Computes mean squared error between two flow fields.

    Args:
        flow1 (tensor):
            flow field, which must have the same dimension as flow2
        flow2 (tensor):
            flow field, which must have the same dimension as flow1

    Returns:
        float: scalar average squared end-point-error
    """
    return torch.mean((flow1 - flow2)**2)

def f_epe(pred, target):
    """Wrapper function to compute the average endpoint error between prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar average endpoint error
    """
    return avg_epe(pred, target)


def f_mse(pred, target):
    """Wrapper function to compute the mean squared error between prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar average squared end-point-error
    """
    return avg_mse(pred, target)


def f_cosim(pred, target):
    """Compute the mean cosine similarity between the two flow fields prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar mean cosine similarity
    """
    return -1 * torch.mean(torch.nn.functional.cosine_similarity(pred, target))


def two_norm_avg(x):
    """Computes the L2-norm of the input normalized by the root of the number of elements.

    Args:
        x (tensor):
            input tensor with variable dimensions

    Returns:
        float: normalized L2-norm
    """
    numels_x = torch.numel(x)
    sqrt_numels = numels_x**0.5
    two_norm = torch.sqrt(torch.sum(torch.pow(torch.flatten(x), 2)))
    return two_norm / sqrt_numels


def get_loss(f_type, pred, target):
    """Wrapper to return a specified loss metric.

    Args:
        f_type (str):
            specifies the returned metric. Options: [aee | mse | cosim]
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Raises:
        NotImplementedError: Unknown metric.

    Returns:
        float: scalar representing the loss measured with the specified norm
    """

    similarity_term = None

    if f_type == "aee":
        similarity_term = f_epe(pred, target)
    elif f_type == "cosim":
        similarity_term = f_cosim(pred, target)
    elif f_type == "mse":
        similarity_term = f_mse(pred, target)
    else:
        raise(NotImplementedError, "The requested loss type %s does not exist. Please choose one of 'aee', 'mse' or 'cosim'" % (f_type))

    return similarity_term

def avg_sq_dist(dist):
    return torch.mean(torch.sum(dist**2, dim=-1))

def avg_sq_dist_weighted_depth(dist, depth):
    eps = 1e-7
    return torch.mean(torch.sum(dist**2, dim=-1)/(depth**2+eps))

def mse_transparency_change(transparency, transparency_init):
    tr   = 1./2. * (torch.tanh(transparency) + 1)
    tr_i = 1./2. * (torch.tanh(transparency_init) + 1)
    return torch.mean((tr-tr_i)**2)

def loss_weather(pred, target, f_type="aee", init_pos=None, offsets=None, alph_offsets=50, motion_offsets=None, alph_motion=100, flakes_transp=None, flakes_transp_init=None, alph_transp=10):

    similarity_term = get_loss(f_type, pred, target)

    offset_term = 0.
    motion_offset_term = 0.
    transp_term = 0.

    if init_pos is not None:

        if alph_motion != 0 or alph_offsets != 0:
            depths = init_pos[...,2]
            if alph_offsets != 0:
                offset_term = alph_offsets*avg_sq_dist_weighted_depth(offsets[...,:3], depths)
            if alph_motion != 0:
                motion_offset_term = alph_motion*avg_sq_dist_weighted_depth(motion_offsets[...,:3], depths)
        if alph_transp != 0:
            transp_term = alph_transp*mse_transparency_change(flakes_transp, flakes_transp_init)

    return similarity_term + offset_term + motion_offset_term + transp_term