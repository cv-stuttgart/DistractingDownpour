import json
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
import os
import os.path as osp
import sys
#required to prevent ModuleNotFoundError for 'flow_plot'. The flow_library is a submodule, which imports its own functions and can therefore not be imported with flow_library.flow_plot
sys.path.append("flow_library")


from PIL import Image

from torch.utils.data import DataLoader, Subset
from helper_functions import datasets
from helper_functions.config_specs import Paths, Conf
from flow_plot import colorplot_light



def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


class InputPadder:
    """Pads images such that dimensions are divisible by divisor

    This method is taken from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    """
    def __init__(self, dims, divisor=8, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        """Pad a batch of input images such that the image size is divisible by the factor specified as divisor

        Returns:
            list: padded input images
        """
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def get_dimensions(self):
        """get the original spatial dimension of the image

        Returns:
            int: original image height and width
        """
        return self.ht, self.wd

    def get_pad_dims(self):
        """get the padding dimensions

        Returns:
            int: original image height and width
        """
        return self._pad

    def unpad(self,x):
        """undo the padding and restore original spatial dimension

        Args:
            x (tensor): a tensor with padded dimensions

        Returns:
            tesnor: tensor with removed padding (i.e. original spatial dimension)
        """
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def import_and_load(net='RAFT', make_unit_input=False, variable_change=False, device=torch.device("cpu"), make_scaled_input_weather_model=False, **kwargs):
    """import a model and load pretrained weights for it

    Args:
        net (str, optional):
            the desired network to load. Defaults to 'RAFT'.
        make_unit_input (bool, optional):
            model will assume input images in range [0,1] and transform to [0,255]. Defaults to False.
        variable_change (bool, optional):
            apply change of variables (COV). Defaults to False.
        device (torch.device, optional):
            changes the selected device. Defaults to torch.device("cpu").
        make_scaled_input_model (bool, optional):
            load a scaled input model which uses make_unit_input and variable_change as specified. Defaults to False.

    Raises:
        RuntimeWarning: Unknown model type

    Returns:
        torch.nn.Module: PyTorch optical flow model with loaded weights
    """

    path_weights = ""

    if make_scaled_input_weather_model:
        from helper_functions.own_models import ScaledInputWeatherModel
        model = ScaledInputWeatherModel(net, make_unit_input=make_unit_input, variable_change=variable_change, device=device, **kwargs)
        print("--> transforming model to 'make_unit_input'=%s, 'make_scaled_input_weather_model'==%s\n" % (str(make_unit_input), str(make_scaled_input_weather_model)))
        path_weights = model.return_path_weights()

    else:
        model = None
        path_weights = ""
        custom_weight_path = kwargs["custom_weight_path"] if "custom_weight_path" in kwargs else ""
        try:
            if net == 'RAFT':
                from models.raft.raft import RAFT

                # set the path to the corresponding weights for initializing the model
                path_weights = custom_weight_path or 'models/_pretrained_weights/raft-sintel.pth'

                # possible adjustements to the config can be made in the file
                # found under models/_config/raft_config.json
                with open("models/_config/raft_config.json") as file:
                    config = json.load(file)

                model = torch.nn.DataParallel(RAFT(config))
                # load pretrained weights
                model.load_state_dict(torch.load(path_weights, map_location=device))

            elif net == 'GMA':
                from models.gma.network import RAFTGMA

                # set the path to the corresponding weights for initializing the model
                path_weights = custom_weight_path or 'models/_pretrained_weights/gma-sintel.pth'

                # possible adjustements to the config file can be made
                # under models/_config/gma_config.json
                with open("models/_config/gma_config.json") as file:
                    config = json.load(file)
                    # GMA accepts only a Namespace object when initializing
                    config = Namespace(**config)

                model = torch.nn.DataParallel(RAFTGMA(config))

                model.load_state_dict(torch.load(path_weights, map_location=device))

            elif net == "FlowFormer":
                from models.FlowFormer.core.FlowFormer import build_flowformer
                from models.FlowFormer.configs.things_eval import get_cfg as get_things_cfg

                path_weights = custom_weight_path or 'models/_pretrained_weights/flowformer_weights/sintel.pth'
                cfg = get_things_cfg()
                model_args = Namespace(model=path_weights, mixed_precision=False, alternate_corr=False)
                cfg.update(vars(model_args))

                model = torch.nn.DataParallel(build_flowformer(cfg))
                model.load_state_dict(torch.load(cfg.model, map_location=torch.device('cpu')))

            elif net =='SpyNet':
                from models.SpyNet.SpyNet import Network as SpyNet
                # weights for SpyNet are loaded during initialization
                model = SpyNet(nlevels=6, pretrained=True)
                model.to(device)

            elif net[:8] == "FlowNet2":
                # hard coding configuration for FlowNet2
                args_fn = Namespace(fp16=False, rgb_max=255.0)

                if net == "FlowNet2":
                    from models.FlowNet.FlowNet2 import FlowNet2
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/FlowNet2_checkpoint.pth.tar'
                    model = FlowNet2(args_fn, div_flow=20, batchNorm=False)
                else:
                    raise ValueError("Unknown FlowNet2 type: %s" % (net))


                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights['state_dict'])

                model.to(device)

            elif net == "FlowNetCRobust":
                from models.FlowNetCRobust.FlowNetC_flexible_larger_field import FlowNetC_flexible_larger_field

                # initialize model and load pretrained weights
                path_weights = custom_weight_path or 'models/_pretrained_weights/RobustFlowNetC.pth'
                model = FlowNetC_flexible_larger_field(kernel_size=3, number_of_reps=3)
                
                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights)

                model.to(device)

            elif net[:7] == "FlowNet":
                # hard coding configuration for FlowNet
                args_fn = Namespace(fp16=False, rgb_max=255.0)

                if net == "FlowNetC":
                    from models.FlowNet.FlowNetC import FlowNetC
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/flownetc_EPE1.766.tar'
                    model = FlowNetC(args_fn, div_flow=20, batchNorm=False)

                else:
                    raise ValueError("Unknown FlowNet type: %s" % (net))

                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights['state_dict'])

                model.to(device)

            if model is None:
                raise RuntimeWarning('The network %s is not a valid model option for import_and_load(network). No model was loaded. Use "RAFT", "GMA", "FlowNetC", "PWCNet" or "SpyNet" instead.' % (net))
        except FileNotFoundError as e:
            print("\nLoading the model failed, because the checkpoint path was invalid. Are the checkpoints placed in models/_pretrained_weights/? If this folder is empty, consider to execute the checkpoint loading script from scripts/load_all_weights.sh. The full error that caused the loading failure is below:\n\n%s" % e)
            exit()

        print("--> flow network is set to %s" % net)
    return model, path_weights

def prepare_dataloader(args, shuffle=False, batch_size=1, get_weather=False):
    """Get a PyTorch dataloader for the specified dataset
    """

    if args.dataset == 'Sintel':
        if args.dataset_stage == 'training':
            dataset = datasets.MpiSintel(args, split=Paths.splits("sintel_train"),
                root=Paths.config("sintel_mpi"), has_gt=True, get_weather=get_weather)
        elif args.dataset_stage == 'evaluation':
            # with this option, ground truth and valid are None!!
            dataset = datasets.MpiSintel(args, split=Paths.splits("sintel_eval"),
                root=Paths.config("sintel_mpi"), has_gt=False, get_weather=get_weather)
        else:
            raise ValueError(f'The specified mode: {args.dataset_stage} is unknown.')
    else:
        raise ValueError("Unknown dataset %s, use 'Sintel'." %(args.dataset))

    # if e.g. the evaluation dataset does not provide a ground truth this is specified
    ds_has_gt = dataset.has_groundtruth()
    ds_has_cam = dataset.has_cameradata()
    ds_has_weat = dataset.has_weatherdata()

    if args.small_run:
        reduced_num_samples = 32
        rand_indices = np.random.randint(0, len(dataset), reduced_num_samples)
        indices = np.arange(0, reduced_num_samples)
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), ds_has_gt, ds_has_cam, ds_has_weat


def preprocess_img(network, *images):
    """Manipulate input images, such that the specified network is able to handle them

    Args:
        network (str):
            Specify the network to which the input images are adapted

    Returns:
        InputPadder, *tensor:
            returns the Padder object used to adapt the image dimensions as well as the transformed images
    """
    if network in ['RAFT', 'GMA', 'FlowFormer']:
        padder = InputPadder(images[0].shape)
        output = padder.pad(*images)

    elif network == 'SpyNet':
        # normalize images to [0, 1]
        images = [ img / 255. for img in images ]
        # make image divisibile by 64
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    elif network[:7] == 'FlowNet':
        # normalization only for FlowNet, not FlowNet2
        if not network[:8] == 'FlowNet2':
            images = [ img / 255. for img in images ]
        # make image divisibile by 64
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    else:
        padder = None
        output = images

    return padder, output


def postprocess_flow(network, padder, *flows):
    """Manipulate the output flow by removing the padding

    Args:
        network (str): name of the network used to create the flow
        padder (InputPadder): instance of InputPadder class used during preprocessing
        flows (*tensor): (batch) of flow fields

    Returns:
        *tensor: output with removed padding
    """

    if padder != None:
        # remove padding
        return [padder.unpad(flow) for flow in flows]

    else:
        return flows


def compute_flow(model, network, x1, x2, test_mode=True, **kwargs):
    """subroutine to call the forward pass of the network

    Args:
        model (torch.nn.module):
            instance of optical flow model
        network (str):
            name of the network. [scaled_input_model | RAFT | GMA | FlowNet2 | SpyNet | PWCNet]
        x1 (tensor):
            first image of a frame sequence
        x2 (tensor):
            second image of a frame sequence
        test_mode (bool, optional):
            applies only to RAFT and GMA such that the forward call yields only the final flow field. Defaults to True.

    Returns:
        tensor: optical flow field
    """
    if network == "scaled_input_model":
        flow = model(x1,x2, test_mode=True, **kwargs)

    elif network == "scaled_input_weather_model":
        flow = model(x1,x2, test_mode=True, **kwargs)

    elif network == 'RAFT':
        _, flow = model(x1, x2, test_mode=test_mode, **kwargs)

    elif network == 'GMA':
        _, flow = model(x1, x2, iters=6, test_mode=test_mode, **kwargs)

    elif network == 'FlowFormer':
        flow = model(x1, x2)[0]

    elif network == 'FlowNetCRobust':
        flow = model(x1, x2)

    elif network[:7] == 'FlowNet':
        # all flow net types need image tensor of dimensions [batch, colors, image12, x, y] = [b,3,2,x,y]
        x = torch.stack((x1, x2), dim=-3)
        # FlowNet2-variants: all fine now, input [0,255] is taken.

        if not network[:8] == 'FlowNet2':
            # FlowNet variants need input in [-1,1], which is achieved by substracting the mean rgb value from the image in [0,1]
            rgb_mean = x.contiguous().view(x.size()[:2]+(-1,)).mean(dim=-1).view(x.size()[:2] + (1,1,1,)).detach()
            x = x - rgb_mean

        flow = model(x)

    elif network in ['SpyNet']:
        with warnings.catch_warnings():
            # this will catch the deprecated warning for spynet and pwcnet to avoid messy console
            warnings.filterwarnings("ignore", message="nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
            warnings.filterwarnings("ignore", message="Default upsampling behavior when mode={} is changed")
            warnings.simplefilter("ignore", UserWarning)
            flow = model(x1,x2, **kwargs)

    else:
        flow = model(x1,x2, **kwargs)

    return flow



def model_takes_unit_input(model):
    """Boolean check if a network needs input in range [0,1] or [0,255]

    Args:
        model (str):
            name of the model

    Returns:
        bool: True -> [0,1], False -> [0,255]
    """
    model_takes_unit_input = False
    if model in ["SpyNet", "FlowNetCRobust"]:
        model_takes_unit_input = True
    return model_takes_unit_input

def get_dimensions(data_loader):
    """Return the image dimensions of the first element of a data_loader

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader of an image dataset

    Returns:
        int, int: image height and image width
    """
    temp_img, _, _, _ = next(iter(data_loader))
    img_h, img_w = temp_img.size()[-2:]
    return img_h, img_w

def flow_length(flow):
    """Calculates the length of the flow vectors of a flow field

    Args:
        flow (tensor):
            flow field tensor of dimensions (b,2,H,W) or (2,H,W)

    Returns:
        torch.float: length of the flow vectors f_ij, computed as sqrt(u_ij^2 + v_ij^2) in a tensor of (b,1,H,W) or (1,H,W)
    """
    flow_pow = torch.pow(flow,2)
    flow_norm_pow = torch.sum(flow_pow, -3, keepdim=True)

    return torch.sqrt(flow_norm_pow)


def maximum_flow(flow):
    """Calculates the length of the longest flow vector of a flow field

    Args:
        flow (tensor):
            a flow field tensor of dimensions (b,2,H,W) or (2,H,W)

    Returns:
        float: length of the longest flow vector f_ij, computed as sqrt(u_ij^2 + v_ij^2)
    """
    return torch.max(flow_length(flow)).cpu().detach().numpy()


def quickvis_tensor(t, filename):
    """Saves a tensor with three dimensions as image to a specified file location.

    Args:
        t (tensor):
            3-dimensional tensor, following the dimension order (c,H,W)
        filename (str):
            name for the image to save, including path and file extension
    """
    # check if filename already contains .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    valid = False
    if len(t.size())==3:
        img = t.detach().cpu().numpy()
        valid = True

    elif len(t.size())==4 and t.size()[0] == 1:
        img = t[0,:,:,:].detach().cpu().numpy()
        valid = True

    else:
        print("Encountered invalid tensor dimensions %s, abort printing." %str(t.size()))

    if valid:
        img = np.rollaxis(img, 0, 3)
        data = img.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)




def quickvisualization_tensor(t, filename, min=0., max=255.):
    """Saves a batch (>= 1) of image tensors with three dimensions as images to a specified file location.
    Also rescales the color values according to the specified range of the color scale.

    Args:
        t (tensor):
            batch of 3-dimensional tensor, following the dimension order (b,c,H,W)
        filename (str):
            name for the image to save, including path and file extension. Batches will append a number at the end of the filename.
        min (float, optional):
            minimum value of the color scale used by tensor. Defaults to 0.
        max (float, optional):
            maximum value of the color scale used by tensor Defaults to 255.
    """
    # rescale to [0,255]
    t = (t.detach().clone() - min) / (max - min) * 255.

    if len(t.size())==3 or (len(t.size())==4 and t.size()[0] == 1):
        quickvis_tensor(t, filename)

    elif len(t.size())==4:
        for i in range(t.size()[0]):
            if i == 0:
                quickvis_tensor(t[i,:,:,:], filename)
            else:
                quickvis_tensor(t[i,:,:,:], filename+"_"+str(i))

    else:
        print("Encountered unprocessable tensor dimensions %s, abort printing." %str(t.size()))


def quickvis_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a flow field tensor with two dimensions as image to a specified file location.

    Args:
        flow (tensor):
            2-dimensional tensor (c=2), following the dimension order (c,H,W) or (1,c,H,W)
        filename (str):
            name for the image to save, including path and file extension.
        auto_scale (bool, optional):
            automatically scale color values. Defaults to True.
        max_scale (int, optional):
            if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    # check if filename already contains .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    valid = False
    if len(flow.size())==3:
        flow_img = flow.clone().detach().cpu().numpy()
        valid = True

    elif len(flow.size())==4 and flow.size()[0] == 1:
        flow_img = flow[0,:,:,:].clone().detach().cpu().numpy()
        valid = True

    else:
        print("Encountered invalid tensor dimensions %s, abort printing." %str(flow.size()))

    if valid:
        # make directory and ignore if it exists
        if not osp.dirname(filename) == "":
            os.makedirs(osp.dirname(filename), exist_ok=True)
        # write flow
        flow_img = np.rollaxis(flow_img, 0, 3)
        data = colorplot_light(flow_img, auto_scale=auto_scale, max_scale=max_scale, return_max=False)
        data = data.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)


def quickvisualization_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a batch (>= 1) of 2-dimensional flow field tensors as images to a specified file location.

    Args:
        flow (tensor):
            single or batch of 2-dimensional flow tensors, following the dimension order (c,H,W) or (b,c,H,W)
        filename (str):
            name for the image to save, including path and file extension.
        auto_scale (bool, optional):
            automatically scale color values. Defaults to True.
        max_scale (int, optional):
            if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    if len(flow.size())==3 or (len(flow.size())==4 and flow.size()[0] == 1):
        quickvis_flow(flow, filename, auto_scale=auto_scale, max_scale=max_scale)

    elif len(flow.size())==4:
        for i in range(flow.size()[0]):
            if i == 0:
                quickvis_flow(flow[i,:,:,:], filename, auto_scale=auto_scale, max_scale=max_scale)
            else:
                quickvis_flow(flow[i,:,:,:], filename+"_"+str(i), auto_scale=auto_scale, max_scale=max_scale)

    else:
        print("Encountered unprocessable tensor dimensions %s, abort printing." %str(flow.size()))


def torchfloat_to_float64(torch_float):
    """helper function to convert a torch.float to numpy float

    Args:
        torch_float (torch.float):
            scalar floating point number in torch

    Returns:
        numpy.float: floating point number in numpy
    """
    float_val = np.float(torch_float.detach().cpu().numpy())
    return float_val


def get_robust_seed(seq, frame):
    seqhash = sum([ord(i) for i in seq])*100
    result = 1234 + int(frame) + int(seqhash)
    return result








