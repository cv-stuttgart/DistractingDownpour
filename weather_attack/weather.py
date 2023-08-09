import torch
import numpy as np
import os
import imageio
import torchvision
import colorsys

from scipy import ndimage
from scipy.spatial.transform import Rotation

import torch.nn.functional as F
import cv2

from torch.multiprocessing import Pool
from torch import multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)


from .render import threeD_to_twoD, matmul3D
from weather_attack import utils


def scale_flake_alpha(flake, depth, args):
    # Attention: only scales alpha channel of flake, not the RGB channels
    # Because scaling the RGB channels might introduce dark artifacts into the color channels during padding
    # Create particle, scale particles by 1/depth
    scaling = 1./ depth / args.depth_decay
    scaling = min(1., scaling) # the min(1,scale) make sure that flake not scaled larger than template size
    scaling = scaling if scaling > 1./args.flakesize_max else 1./args.flakesize_max # check if scaling retains a minimal size of 1 px, otherwise set to 1 px.
    flake_alph_scaled = F.interpolate(torch.unsqueeze(flake[...,3,:,:],-3), scale_factor=scaling, mode='bilinear')
    newdim = flake_alph_scaled.size(-1)
    flake[...,3,:,:] = F.pad(flake_alph_scaled, (int(np.floor((args.flakesize_max-newdim)/2.)),int(np.ceil((args.flakesize_max-newdim)/2.)),int(np.floor((args.flakesize_max-newdim)/2.)),int(np.ceil((args.flakesize_max-newdim)/2.))), "constant", 0)
    return flake


def initialize_transparency(depth):
    # Set particle transparency depending on depth before tanh trafo (-2->0.018, 0->0.5, 2->0.98)
    transp = 4/3 * depth - 1.5 if depth < 1.5 else -4/3 * depth + 2.5
    return torch.tensor(transp)[None]


def init_flake(args, rng=None, flake_array=None):

    flake_color = torch.ones(1, 3)
    H,L,S = colorsys.rgb_to_hls(args.flake_r/255., args.flake_g/255., args.flake_b/255)
    # Calculate randomization in HLS
    # random*(upper_bound - lower_bound)
    if rng is None:
        rdh = 2*np.random.rand(1)[0]-1.
        rdl = 2*np.random.rand(1)[0]-1.
        rds = 2*np.random.rand(1)[0]-1.
    else:
        rdh = 2*rng.random(1)[0]-1.
        rdl = 2*rng.random(1)[0]-1.
        rds = 2*rng.random(1)[0]-1.
    dh = (rdh*args.flake_random_h % 360) / 360. # 0-1
    dl = rdl*args.flake_random_l
    ds = rds*args.flake_random_s
    # Add HLS offsets and return HLS value to RGB
    R,G,B = colorsys.hls_to_rgb(H+dh, np.clip([L+dl],0.,1.)[0], np.clip([S+ds],0.,1.)[0])
    flake_color[0] = torch.tensor([R,G,B])

    if not args.do_shape:
        flake = torch.ones(1, 4, args.flakesize_max, args.flakesize_max)
    else:
        if rng is None:
            flake_dat = np.random.choice(flake_array).copy()
        else:
            flake_dat = rng.choice(flake_array).copy()

        if rng is None:
            ang = np.random.rand(1)[0]*360.
        else:
            ang = rng.random(1)[0]*360.
        if args.do_rotate:
            flake_rotate = ndimage.rotate(flake_dat, ang, reshape=False)
        else:
            flake_rotate = flake_dat
        # flake_rotate = rotate(torch.tensor(flake_dat), ang, fill=0)
        # take first channel only, and expand to (1,3,h,w) tensor to allow for bilinear interpolation
        # flake_large = torch.tensor(flake_dat, dtype=float).permute((2, 0, 1)).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        flake_large = torch.from_numpy(flake_rotate.astype(np.float32)).permute((2, 0, 1))
        # interpolate flake image to requested flake size
        flake = flake_large.unsqueeze(0)
        # padd flake image with transparent border of size pd, and use reflecting boundaries for the color channels. Then interpolate to new size.
        pd = 1
        flake_padded = F.pad(flake,(pd,pd,pd,pd),mode='reflect')
        flake_alpha_padded = F.pad(flake[...,3,:,:],(pd,pd,pd,pd),mode='constant', value=0.)
        flake_padded[...,3,:,:] = flake_alpha_padded
        flake =  flake_padded
        flake = F.interpolate(flake, scale_factor=args.flakesize_max/(flake_large.size(-1)+2*pd), mode='bilinear')

    return flake, flake_color


def blur_flake_alpha(flake, depth, args):
    b = 0.563
    a = 5 * (0.75-b)
    size = int(np.round(np.clip(a / (depth-b), a_min=0, a_max=20)))
    if size <= 1:
        return flake
    return blur(flake, size)


def blur(input_img, size=3):
    # assert batch size 1
    assert input_img.shape[0] == 1

    input_img = input_img[0].cpu().numpy()

    output = []
    for channel in range(input_img.shape[0]):
        if channel in [3]:

            img = input_img[channel]
            img = img * 2
            # do dft saving as complex output
            dft_img = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

            # create circle mask
            mask = np.zeros_like(img)
            cy = mask.shape[0] // 2
            cx = mask.shape[1] // 2
            cv2.circle(mask, (cx,cy), size, 1, -1)[0]

            # blur the mask slightly to antialias
            mask = cv2.GaussianBlur(mask, (3,3), 0)

            centerx = img.shape[1] // 2
            centery = img.shape[0] // 2
            # roll the mask so that center is at origin and normalize to sum=1
            mask_roll = np.roll(mask, (centerx,centery), axis=(0,1))
            mask_norm = mask_roll / mask_roll.sum()

            # take dft of mask
            dft_mask_norm = cv2.dft(np.float32(mask_norm), flags = cv2.DFT_COMPLEX_OUTPUT)

            # apply dft_mask to dft_img
            dft_product = cv2.mulSpectrums(dft_img, dft_mask_norm, 0)

            # do idft saving as complex output, then clip and convert to uint8
            img_filtered = cv2.idft(dft_product, flags=cv2.DFT_SCALE+cv2.DFT_REAL_OUTPUT)
            img_filtered = img_filtered.clip(0,5)
            img_filtered = img_filtered ** (1/3)
            img_filtered = img_filtered.clip(0,1)

            output.append(img_filtered)
        else:
            img = input_img[channel]
            output.append(img)
    output = torch.from_numpy(np.asarray(output))[None]
    return output

def load_flake_dat(flake_path):
    flake_alpha = imageio.imread(flake_path)[:,:,0] # read R-channel as alpha
    flake_dat = np.stack((np.zeros_like(flake_alpha), np.zeros_like(flake_alpha), np.zeros_like(flake_alpha), flake_alpha), axis=2) / 255.
    return flake_dat

def extract_flakes(args):
    if not args.do_shape:
        flake_array = [np.ones((4, args.flakesize_max, args.flakesize_max))]
    else:
        f_path = os.path.join(args.flake_folder, args.flake_template_folder)
        flake_files = [f for f in os.listdir(f_path) if os.path.isfile(os.path.join(f_path, f)) and os.path.splitext(os.path.join(f_path, f))[-1]==".png"]
        flake_files = np.sort(flake_files)

        flake_array = [load_flake_dat(os.path.join(f_path, i)) for i in flake_files]
    return np.array(flake_array)


def get_weather(has_weather, weatherdat, scene_data, args, seed=None, load_only=False):
    if has_weather:
        print("Loading existing weather model instead of generating particles.")
        weather, success = utils.load_weather(weatherdat)
        if success:
          if args.recolor:
            weather = recolor_weather(weather, args)
          weather = [torch.from_numpy(i) for i in weather]
    if load_only and not success:
      raise ValueError(f"No weather data was found at {weatherdat}")
    if not has_weather or not success:
        weather = generate_weather(scene_data, args, seed=seed)
    return weather


def recolor_weather(weather, args):
    (points3D, motion3D, flakes, flakescol, flakestrans) = weather
    flakescol[:,:,0] = np.ones_like(flakescol[:,:,0])*args.flake_r/255.
    flakescol[:,:,1] = np.ones_like(flakescol[:,:,1])*args.flake_g/255.
    flakescol[:,:,2] = np.ones_like(flakescol[:,:,2])*args.flake_b/255.

    return (points3D, motion3D, flakes, flakescol, flakestrans)


def generate_weather(scene_data, args, seed=None):

    if args.cpu_count == 0:
        cpucount = multiprocessing.cpu_count() // 2
    else:
        cpucount = args.cpu_count

    assert args.flakesize_max % 2 != 0

    points3D_b = []
    motion3D_b = []
    flakes_b = []
    flakescol_b = []
    flakestrans_b = []

    # fist create more particles than needed
    split_size = args.num_flakes // cpucount + 1

    flake_array = extract_flakes(args)

    if seed is None:
        arguments = cpucount * [(scene_data, split_size, args, seed, flake_array)]
    else:
        arguments = [(scene_data, split_size, args, seed+i, flake_array) for i in range(cpucount)]

    result = []
    with Pool(cpucount) as p:
        result = p.starmap(initialize_weather_split, arguments)
    points3D = []
    motion3D = []
    flakes = []
    flakescol = []
    flakestrans = []
    for p, m, f, fc, ft in result:
        points3D.append(p)
        motion3D.append(m)
        flakes.append(f)
        flakescol.append(fc)
        flakestrans.append(ft)

    # now take away unnecessary particles [:,:num_flakes]
    points3D_b = torch.cat(points3D, dim=1)[:,:args.num_flakes]
    flakes_b = torch.cat(flakes, dim=1)[:,:args.num_flakes]
    flakescol_b = torch.cat(flakescol, dim=1)[:,:args.num_flakes]
    flakestrans_b = torch.cat(flakestrans, dim=1)[:,:args.num_flakes]
    motion3D_b = torch.cat(motion3D, dim=1)[:,:args.num_flakes]

    return points3D_b, motion3D_b, flakes_b, flakescol_b, flakestrans_b


def initialize_weather_split(scene_data, split_size, args, seed=None, flake_array=None):

    full_P, ext1, rel_mat, gt1_depth, gt2_depth = scene_data

    rng = None
    if seed is None:
        rng = np.random.default_rng(multiprocessing.current_process().pid)
    else:
        rng = np.random.default_rng(seed)

    #print(ext1)
    r = Rotation.from_matrix(ext1[0,:3,:3].numpy())
    euler_rot = r.as_euler("xyz", degrees=True)
    xrot = euler_rot[0]
    # convert x rotation:
    xrot_ = 180 - np.abs(xrot)
    # weather should fall down, not in cam direction
    xrot_ -= 90
    rotmat = Rotation.from_euler("xyz", [xrot_, 0, 0], degrees=True)
    rotmat = np.vstack((np.hstack((rotmat.as_matrix(), [[0],[0],[0]])), [[0,0,0,1]]))

    points3D_b = []
    motions3D_b = []
    flakes_b = []
    flakes_color_b = []
    flakes_transp_b = []

    for b in range(gt1_depth.size(0)):
        max_gt1_depth = gt1_depth[b].max()
        max_gt2_depth = gt2_depth[b].max()

        points3D = []
        motions3D = []
        flakes = []
        flakes_color = []
        flakes_transp = []

        while True:

            p3d_rd = rng.random((1,1,3), dtype=np.float32)
            point3d = torch.tensor(p3d_rd, device=full_P.device)
            point3d[:,:,:2] = 10 * point3d[:,:,:2] - 5
            point3d[:,:,2] = 3 * point3d[:,:,2]
            point3d = torch.cat((point3d, torch.ones((1,1,1))), dim=-1)

            # Initialize motion vector
            motion3d = torch.tensor([args.motion_x, args.motion_y, args.motion_z,1])[None,None,:] # shape: (1,1,4)

            # Randomize motion angle by creating a rotation matrix with random angles in the allowed range
            rd_dir = (rng.random((3), dtype=np.float32) - .5) * 2. * args.motion_random_angle
            rd_rot = Rotation.from_euler('xyz', [rd_dir[0], rd_dir[1], rd_dir[2]], degrees=True)
            rd_rotmat = np.vstack((np.hstack((rd_rot.as_matrix(), [[0],[0],[0]])), [[0,0,0,1]]))
            motion3d[0,0] = motion3d[0,0] @ rd_rotmat

            # Randomize the vector scale by the allowed scaling factor
            rd_scale = 1 + (rng.random((1), dtype=np.float32) - 0.5) * 2. * args.motion_random_scale
            scale = torch.tensor([rd_scale[0],rd_scale[0],rd_scale[0],0.], device=full_P.device)
            motion3d = motion3d * scale

            # Apply global coordinate system transform (particle falls relative to world coordinates, not camera plane)
            motion3d[0,0] = motion3d[0,0] @ rotmat

            accept_point = False
            accept_frame = -1 # logg in which frame the point was accepted (for scaling / transparency)

            if point3d[:,:,2] < max_gt1_depth:
                point2d, point_depth = threeD_to_twoD(full_P, point3d)
                point2d = torch.round(point2d).long()

                # accept point if it is in first frame
                if point2d[...,0] >= 0 and point2d[...,0] < gt1_depth.size(-1) and point2d[...,1] >= 0 and point2d[...,1] < gt1_depth.size(-2):
                    # if np.abs(ggt1_depth[b,0,point2d[...,1].long(), point2d[...,0].long()] - point_depth[...,0]) < 0.1:
                    if point_depth[...,0] < gt1_depth[b,0,point2d[...,1], point2d[...,0]]:
                        # Append point
                        accept_point = True
                        accept_frame = 0

            # if point not in first frame, check if it is in second frame and accept if so.
            if not accept_point:
                point3d_moved = matmul3D(rel_mat, point3d) + motion3d
                if point3d_moved[:,:,2] < max_gt2_depth:
                    point2d_moved, point_depth_moved = threeD_to_twoD(full_P, point3d_moved)
                    # point2d_moved = np.round(point2d_moved).astype(int)
                    point2d_moved = torch.round(point2d_moved).long()
                    if point2d_moved[...,0] >= 0 and point2d_moved[...,0] < gt2_depth.size(-1) and point2d_moved[...,1] >= 0 and point2d_moved[...,1] < gt2_depth.size(-2):
                        # if np.abs(gt2_depth[b,0,point2d_moved[...,1].long(), point2d_moved[...,0].long()] - point_depth_moved[...,0].long()) < 0.1:
                        if point_depth_moved[...,0] < gt2_depth[b,0,point2d_moved[...,1], point2d_moved[...,0]]:
                            # Append point
                            accept_point = True
                            accept_frame = 1

            if accept_point:
                points3D.append(point3d[0])
                motions3D.append(motion3d[0])

                # set point depth to depth in a frame that accepted the point
                if accept_frame == 0:
                    depth = point_depth[0,0].item()
                elif accept_frame == 1:
                    depth = point_depth_moved[0,0].item()

                flake, flake_color = init_flake(args, rng=rng, flake_array=flake_array)
                if args.do_scale:
                    flake = scale_flake_alpha(flake, depth, args)
                if args.do_blur:
                    flake = blur_flake_alpha(flake, depth, args)

                if args.do_transp:
                    flake_transp = initialize_transparency(depth)
                else:
                    flake_transp = torch.tensor(1000.)[None]

                flakes.append(flake)
                flakes_color.append(flake_color)
                flakes_transp.append(flake_transp)

                if len(points3D) >= split_size:
                    break

        points3D = torch.cat(points3D, dim=0)[None]
        motions3D = torch.cat(motions3D, dim=0)[None]
        flakes = torch.cat(flakes, dim=0)[None]
        flakes_color = torch.cat(flakes_color, dim=0)[None]
        flakes_transp = torch.cat(flakes_transp, dim=0)[None]

        points3D_b.append(points3D)
        motions3D_b.append(motions3D)
        flakes_b.append(flakes)
        flakes_color_b.append(flakes_color)
        flakes_transp_b.append(flakes_transp)


    points3D_b = torch.cat(points3D_b, dim=0).to(gt1_depth.device)
    motions3D_b = torch.cat(motions3D_b, dim=0).to(gt1_depth.device)
    flakes_b = torch.cat(flakes_b, dim=0).to(gt1_depth.device)
    flakes_color_b = torch.cat(flakes_color_b, dim=0).to(gt1_depth.device)
    flakes_transp_b = torch.cat(flakes_transp_b, dim=0).to(gt1_depth.device)

    flakes_b = flakes_b.float()

    return points3D_b, motions3D_b, flakes_b, flakes_color_b, flakes_transp_b
