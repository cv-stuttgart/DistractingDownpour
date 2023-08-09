# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from helper_functions import frame_utils
from helper_functions.config_specs import Paths
from helper_functions.sintel_io import cam_read, depth_read


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.sparse = sparse

        self.has_gt = False
        self.has_camdata = False
        self.has_weathdata = False
        self.get_weather = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.weather_img = []
        self.extra_info = []
        self.enforce_dimensions = False
        self.image_x_dim = 0
        self.image_y_dim = 0

        # camera data
        self.full_p = []
        self.ext1 = []
        self.rel_mat = []
        self.gt1_depth = []
        self.gt2_depth = []

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        if self.has_weathdata and self.get_weather:
            try:
                img_weather1 = frame_utils.read_gen(self.weather_img[index][0])
                img_weather2 = frame_utils.read_gen(self.weather_img[index][1])
                img_weather1 = np.array(img_weather1).astype(np.uint8)
                img_weather2 = np.array(img_weather2).astype(np.uint8)
            except FileNotFoundError as e:
                img_weather1 = np.zeros_like(img1,dtype=np.uint8)
                img_weather2 = np.zeros_like(img2,dtype=np.uint8)
        extra_info = self.extra_info[index]
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
            if self.has_weathdata and self.get_weather:
                img_weather1 = np.tile(img_weather1[...,None], (1, 1, 3))
                img_weather2 = np.tile(img_weather2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            if self.has_weathdata and self.get_weather:
                img_weather1 = img_weather1[..., :3]
                img_weather2 = img_weather2[..., :3]

        valid = None

        if self.has_gt:
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        else:
            (img_x, img_y, img_chann) = img1.shape

            flow = np.zeros((img_x, img_y, 2)) # make correct size for flow (2 dimensions for u,v instead of 3 [r,g,b] for image )
            valid = False

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.has_camdata and self.get_weather:
            full_p = self.full_p[index]
            ext1 = self.ext1[index]
            rel_mat = self.rel_mat[index]
            gt1_depth = self.gt1_depth[index]
            gt2_depth = self.gt2_depth[index]
        else:
            full_p = 0
            ext1 = 0
            rel_mat = 0
            gt1_depth = 0
            gt2_depth = 0

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        if self.has_weathdata and self.get_weather:
            img_weather1 = torch.from_numpy(img_weather1).permute(2, 0, 1).float()
            img_weather2 = torch.from_numpy(img_weather2).permute(2, 0, 1).float()

        if self.enforce_dimensions:
            dims = img1.size()
            x_dims = dims[-2]
            y_dims = dims[-1]

            diff_x = self.image_x_dim - x_dims
            diff_y = self.image_y_dim - y_dims

            img1 = F.pad(img1, (0,diff_y,0,diff_x), "constant", 0)
            img2 = F.pad(img2, (0,diff_y,0,diff_x), "constant", 0)
            if self.has_weathdata and self.get_weather:
                img_weather1 = F.pad(img_weather1, (0,diff_y,0,diff_x), "constant", 0)
                img_weather2 = F.pad(img_weather2, (0,diff_y,0,diff_x), "constant", 0)

            flow = F.pad(flow, (0,diff_y,0,diff_x), "constant", 0)
            if self.has_gt:
                valid = F.pad(valid, (0,diff_y,0,diff_x), "constant", False)


        if self.has_weathdata and self.get_weather:
            return img1, img2, img_weather1, img_weather2, flow, valid, (full_p, ext1, rel_mat, gt1_depth, gt2_depth), extra_info
        elif self.get_weather:
            return img1, img2, flow, valid, (full_p, ext1, rel_mat, gt1_depth, gt2_depth), extra_info
        else:
            return img1, img2, flow, valid


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

    def has_groundtruth(self):
        return self.has_gt

    def has_cameradata(self):
        return self.has_camdata

    def has_weatherdata(self):
        return self.has_weathdata


def rescale_sintel_scenes(scene, gt1_depth, gt2_depth, rel_mat, OVERALL_SCALE=1., transl_scale=0.):

    GLOBAL_SCALE = 1.0
    # for mountain_1 --> scale 3D scene globally
    if scene in ["alley_1", "alley_2", "ambush_4", "bandage_2", "market_5", "market_6"]:
        GLOBAL_SCALE = 2.
    elif scene in ["mountain_1"]:
        GLOBAL_SCALE = 0.05
    elif scene in ["market_2"]:
        GLOBAL_SCALE = 0.2
    elif scene in ["cave_2"]:
        GLOBAL_SCALE = 0.4
    elif scene in ["ambush_2", "ambush_6"]:
        GLOBAL_SCALE = 1.5
    elif scene in ["shaman_2"]:
        GLOBAL_SCALE = 2.3
    elif scene in ["bandage_1", "shaman_3"]:
        GLOBAL_SCALE = 2.5
    elif scene in ["sleeping_1"]:
        GLOBAL_SCALE = 5.
    elif scene in ["ambush_7"]:
        GLOBAL_SCALE = 7.

    if GLOBAL_SCALE != 1.0 or OVERALL_SCALE != 1.0 or transl_scale != 0:
        gt1_depth *= GLOBAL_SCALE*OVERALL_SCALE
        gt2_depth *= GLOBAL_SCALE*OVERALL_SCALE
        rel_mat[...,:3,3] *= GLOBAL_SCALE*OVERALL_SCALE*transl_scale

    return gt1_depth, gt2_depth, rel_mat


class MpiSintel(FlowDataset):
    def __init__(self, args, aug_params=None, split=Paths.splits("sintel_train"), root=Paths.config("sintel_mpi"), has_gt=False, get_weather=False):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, args.dstype)
        cam_left_root = osp.join(root, split, 'camdata_left')
        depth_root = osp.join(root, split, 'depth')

        self.has_gt = has_gt
        self.has_camdata = has_gt
        try:
          self.has_weathdata = args.weather_data != ""
        except AttributeError:
          self.has_weathdata = False
        self.get_weather = get_weather

        # required if args.from_scene is used to identify scenes after the given scene
        all_scenes_from_here = False
        all_sc = args.single_scene == '' and args.from_scene == ''


        for scene in sorted(os.listdir(image_root)):
            if all_sc or (all_scenes_from_here or scene in [args.single_scene, args.from_scene]):
                if scene == args.from_scene:
                    all_scenes_from_here = True
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                # check if dataset should be shortened
                if args.frame_per_scene == 0:
                    no_imges = len(image_list)-1
                else:
                    no_imges = args.frame_per_scene
                for i in range(len(image_list)-1):
                    if i >= args.from_frame and i < args.from_frame+no_imges:
                      self.image_list += [ [image_list[i], image_list[i+1]] ]
                      weather = ""
                      if self.has_weathdata and self.get_weather:
                          weather = osp.join(args.weather_data, f"{scene}/frame_{i+1:04d}.npz")
                          self.weather_img += [ [osp.join(args.weather_data, f"{scene}/frame_1_{i+1:04d}.png"), osp.join(args.weather_data, f"{scene}/frame_2_{i+1:04d}.png")] ]
                          self.extra_info += [ (root, split, scene, "frame_", i, weather) ] # scene and frame_id
                      elif self.get_weather:
                          self.extra_info += [ (root, split, scene, "frame_", i, "") ] # scene and frame_id
                      else:
                          self.extra_info += [ (scene, i) ] # scene and frame_id

                      if self.has_camdata and self.get_weather:
                          a_int, a_ext_1 = cam_read(os.path.join(cam_left_root, f"{scene}/frame_{i+1:04d}.cam"))
                          _, a_ext_2 = cam_read(os.path.join(cam_left_root, f"{scene}/frame_{i+2:04d}.cam"))
                          a_ext_1 = np.vstack((a_ext_1, np.array([[0,0,0,1]])))
                          a_ext_2 = np.vstack((a_ext_2, np.array([[0,0,0,1]])))

                          full_p = np.vstack((np.hstack((a_int, [[0],[0],[0]])), [[0,0,0,1]]))
                          full_p = torch.from_numpy(full_p).float()
                          rel_mat = torch.from_numpy(a_ext_2 @ np.linalg.inv(a_ext_1)).float()

                          gt1_depth = depth_read(os.path.join(depth_root, f"{scene}/frame_{i+1:04d}.dpt"))
                          gt1_depth = torch.unsqueeze(torch.from_numpy(gt1_depth), dim=0)
                          gt2_depth = depth_read(os.path.join(depth_root, f"{scene}/frame_{i+2:04d}.dpt"))
                          gt2_depth = torch.unsqueeze(torch.from_numpy(gt2_depth), dim=0)

                          gt1_depth, gt2_depth, rel_mat = rescale_sintel_scenes(scene, gt1_depth, gt2_depth, rel_mat, OVERALL_SCALE=args.scene_scale)

                          self.full_p += [full_p]
                          self.ext1 += [a_ext_1]
                          self.rel_mat += [rel_mat]
                          self.gt1_depth += [gt1_depth]
                          self.gt2_depth += [gt2_depth]

                if split != 'test':
                    self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))[args.from_frame:args.from_frame+no_imges]

        if len(self.image_list) == 0:
            raise RuntimeWarning("No MPI Sintel data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the MPI Sintel dataset." % root)
