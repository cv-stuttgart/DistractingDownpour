import torch
import torch.nn.functional as F


def add_things_on_img(img, points, what):
    assert len(img.shape) == 3
    assert img.shape[-1] == 3
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert len(what.shape) == 2
    assert what.shape[1] == 3
    assert what.shape[0] == points.shape[0]

    h,w,_ = img.shape
    valid = (points[:,0] >= 0) & (points[:,0] < w) & ( points[:,1] >= 0) & (points[:,1] < h)
    indices = points[:,0] + w * points[:,1]

    # remove invalid points:
    indices = indices[valid]
    what = what[valid]

    indices = indices.unsqueeze(1).repeat(1,3)
    img_flat = img.reshape(-1,3)
    img_flat.scatter_add_(0, indices, what)
    img2 = img_flat.reshape(h,w,3)
    return img2

def scat_mult(img, points, factors):
    assert len(img.shape) == 3
    assert img.shape[-1] == 3
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert len(factors.shape) == 2
    assert factors.shape[1] == 3
    assert factors.shape[0] == points.shape[0]

    h,w,_ = img.shape
    valid = (points[:,0] >= 0) & (points[:,0] < w) & ( points[:,1] >= 0) & (points[:,1] < h)
    indices = points[:,0] + w * points[:,1]

    # remove invalid points:
    indices = indices[valid]
    factors = factors[valid]

    indices = indices.unsqueeze(1).repeat(1,3)
    img_flat = img.reshape(-1,3)
    img_flat = img_flat.scatter_reduce(0,indices,factors, reduce='prod') # the backward pass for .scatter_()/.scatter() (in/out of place) is not implemented!
    img2 = img_flat.reshape(h,w,3)
    return img2


def matmul3D(mat, tensor):
    """compute matrix multiplication mat @ vec for every vec in tensor - batched"""
    return torch.einsum("jik,jlk->jil", tensor, mat)


def bilinear_interpolation(d0, d1, V00, V01, V10, V11, transp=None):
    if transp is None:
        transp = torch.ones_like(d1[...,0], device=d1.device)
    return d1[...,0]*d1[...,1]*V00*transp + d0[...,1]*d1[...,0]*V01*transp + d0[...,0]*d1[...,1]*V10*transp + d0[...,0]*d0[...,1]*V11*transp


def bilinear_interpolation_new(d0, d1, V00, V01, V10, V11):
    return d1[:,:,0]*d1[:,:,1]*V00 + d0[:,:,1]*d1[:,:,0]*V01 + d0[:,:,0]*d1[:,:,1]*V10 + d0[:,:,0]*d0[:,:,1]*V11


def threeD_to_twoD(full_P, points_3d):
    result_p = matmul3D(full_P, points_3d)
    result = result_p[...,:3]
    depths = result[...,2]
    return result[...,:2] / depths[...,None], depths


def depth_weight(depth):
    return torch.pow(depth, -4)

def twoD_to_img(image, image_depth, point2d, flakes, flakes_color, flakes_transp, flakes_depth, args):

    # nearest upper left pixel position
    V1 = torch.floor(point2d).long()

    # weights for interpolation
    d1 = point2d - V1
    d0 = torch.ones_like(point2d, device=point2d.device) - d1

    # B, flakecount, 3, flake_h, flake_w
    batchsize, _, _, _, flake_size = flakes.shape

    # B, 3, h, w
    _, _, image_h, image_w = image.shape

    # assume flakes to have odd dimensions (3x3, 5x5, ...) -> flake_size = 1 + 2*flake_half
    flake_half = flake_size // 2
    f_p = 2  # flake padding
    flake_pad_half = flake_half+f_p
    img_pad = flake_size+2*f_p

    # Save original image dimensions (for unpadding) and padd
    image = F.pad(image, (img_pad,img_pad,img_pad,img_pad), "constant", 0) # image padding, img_pad because flakes are padded by 2 in every direction
    image_depth = F.pad(image_depth, (img_pad+1,img_pad+1,img_pad+1,img_pad+1), "replicate") # depth padding, 1 larger than image padding because depth needs 1 additonal layer due to depth interpolation to flake position.


    flake_interpol = F.pad(flakes, (f_p,f_p,f_p,f_p), "constant", 0)

    # INTERPOLATION
    flake_interpol_TL = flake_interpol[..., :-1, :-1]
    flake_interpol_TR = flake_interpol[..., :-1, 1: ]
    flake_interpol_BL = flake_interpol[..., 1: , :-1]
    flake_interpol_BR = flake_interpol[..., 1: , 1: ]
    d0 = d0[:,:,:,None,None,None]
    d1 = d1[:,:,:,None,None,None]
    flake_interpol_result = bilinear_interpolation_new(d0,d1,flake_interpol_TL, flake_interpol_TR, flake_interpol_BL, flake_interpol_BR)


    # APPLY CAM CHECK
    # remove flakes that are behind the camera/viewer pos
    cam_check = flakes_depth > 0
    flake_interpol_result *= cam_check[...,None,None,None]


    # DETERMINE TARGET POSITIONS
    i_lower = V1[:,:,0] - (flake_pad_half-1) + img_pad
    j_lower = V1[:,:,1] - (flake_pad_half-1) + img_pad
    y, x = torch.meshgrid(torch.arange(2*flake_pad_half, device=V1.device), torch.arange(2*flake_pad_half, device=V1.device))
    pos = torch.cat((x[None],y[None]),dim=0)

    # DEPTH-AWARENESS
    if args.depth_check:
        # (1) start by creating slice of image_depth in (flakex x flakey) to avoid having depth_map (numflakes x imagex x imagey)
        grid = pos.clone()
        lower = torch.cat((j_lower[...,None] + 1, i_lower[...,None] + 1), dim=-1)
        grid = grid[None,None] + lower[...,None,None]
        grid = grid.permute(0,1,3,4,2).float()
        # normalize grid indices to [-1,1], centered at particle central pixel
        grid[...,0] *= 2.0 / image_depth.shape[3]
        grid[...,1] *= 2.0 / image_depth.shape[2]
        grid -= 1.0
        grid = grid.float()
        # resize to grid (batches, numflakes*flakeX, flakeY, 2)
        gridsize_1 = grid.shape[1] # numflakes
        gridsize_2 = grid.shape[2] # flake_X
        grid = torch.reshape(grid, (grid.shape[0], gridsize_1*gridsize_2 ,grid.shape[3], grid.shape[4]))

        # generate depth_samples of (batches, 1, numflakes*flakeX, flakeY), 1 is the number of depth-channels
        depth_samples = F.grid_sample(image_depth, grid, padding_mode="border")
        depth_samples = torch.reshape(depth_samples, (grid.shape[0], gridsize_1, gridsize_2 ,grid.shape[2]))
        # resizes to (batches, numflakes, flakeX, flakeY)

        ### (2) Now create depth-map with reduced image_depth
        # depth_map = image_depth >= flakes_depth[..., None, None]
        if args.depth_check_differentiable:
          factor = 30
        else:
          factor = 250
        # sharp sigmoid instead of step function - the factor should for differentiability, but large for visual effects.
        flake_depths = 1/(1+torch.exp(factor*(flakes_depth[..., None, None]-depth_samples)))

        if not args.depth_check_differentiable:
            flake_depths = flake_depths.detach()
        # apply depth only to alpha channel
        flake_interpol_result[:,:,3] *= flake_depths

    lower = torch.cat((j_lower[...,None], i_lower[...,None]), dim=-1)
    pos = pos[None,None] + lower[...,None,None]

    # necessary to avoid nans on encountering negative near-zero values (floating inaccuracies that should be zero)
    torch.nn.functional.relu(flake_interpol_result, inplace=True)

    # Split into color and alpha channel (gamma-correction should only affect color)
    flake_interpol_result_color_offsets = flake_interpol_result[...,:3,:,:]
    flake_interpol_result_alpha = torch.unsqueeze(flake_interpol_result[...,3,:,:],2)

    # Then apply alpha to color channels in linear color space
    flake_interpol_result = (flakes_color[...,None,None] + flake_interpol_result_color_offsets) * flake_interpol_result_alpha


    # THIS PART ONLY WORKS WITH BATCH SIZE 1 ==============================

    # img: h x w x 3
    # pos: N x 2
    # flk_xxx: N x 3
    pos = pos.permute(0,1,3,4,2).reshape(batchsize, -1, 2)
    flk_color = flake_interpol_result.permute(0,1,3,4,2).reshape(batchsize, -1, 3)

    # TRANSPARENCIES preparation - alpha is the alpha channel for each flake (creates flake shape) and transparency is the depth-dependent transparency (applied on top)
    flk_transp = args.transparency_scale * 1./2. * (torch.tanh(flakes_transp) + 1) # shape: (batchsize, numflakes)
    flk_transp = torch.ones_like(flake_interpol_result.permute(0,1,3,4,2)) * flk_transp[...,None,None,None]
    flk_transp = flk_transp.reshape(batchsize, -1, 3)
    flk_alpha = torch.ones_like(flake_interpol_result.permute(0,1,3,4,2)) * flake_interpol_result_alpha.permute(0,1,3,4,2)
    flk_alpha = flk_alpha.reshape(batchsize, -1, 3)

    if args.rendering_method == "meshkin":
        # For method details see related work in: https://jcgt.org/published/0002/02/09/
        sum_ai = add_things_on_img(torch.zeros_like(image[0].permute(1,2,0)), pos[0], flk_transp[0]*flk_alpha[0]) # overall transparency = flake_alpha * flake_transp
        sum_ci = add_things_on_img(torch.zeros_like(image[0].permute(1,2,0)), pos[0], flk_transp[0]*flk_color[0]) # part of overall transparency (the alpha) is implicitely in "flk_color" (via flake_interpol_result) and for a full transparency-premultiplication we also need the global transparency attenuation flk_transp.

        image = sum_ci + image[0].permute(1,2,0) * (1 - sum_ai) # Meshkins Method -> needs sum_ci (color*transp) because otherwise (color*transp*alpha) artifacts exist around the particles in the background
    elif args.rendering_method == "additive":
        image = add_things_on_img(image[0].permute(1,2,0), pos[0], flk_transp[0]*flk_color[0])
    else:
        raise RuntimeWarning(f"Unknown transparency method '{args.rendering_method}'. Returning image without particles")
        image = image[0].permute(1,2,0)

    image = image.permute(2,0,1)
    image = image[None]
    # ===========================================================================

    # UNPADDING
    image = image[..., img_pad:image_h+img_pad, img_pad:image_w+img_pad] # image unpadding, flake_size+4 because flakes are padded by 2 in every direction

    return image


def render(image1, image2, scene_data, weather, args):
    """Summary

    Args:
        image1 (TYPE): (B,c,H,W)
        image2 (TYPE): (B,c,H,W)
        full_P (TYPE): Description
        rel_mat (TYPE): Description
        points3D (TYPE): Description
        motion3D (TYPE): Description
        flakes (TYPE): Description

    Returns:
        TYPE: Description
    """

    full_P, _, rel_mat, depth1, depth2 = scene_data
    points3D, motion3D, flakes, flakes_color, flakes_transp = weather

    points3D_moved = matmul3D(rel_mat, points3D) + motion3D

    motion_diff = points3D_moved - points3D

    if args.do_motionblur:
        samples = args.motionblur_samples
    else:
        samples = 1

    image1_result = image1
    image2_result = image2


    for i in range(samples):

        flakes_ = flakes
        flakes_color_ = flakes_color
        # Don't change for motion blur (are always 1D and not dependent on flake-number):
        # full_P, depth1, depth 2

        if samples > 1:
            flakes_transp_ = 1./2. * (torch.tanh(flakes_transp) + 1) / samples # transformation to linear scale and divide by samples
            flakes_transp_ = torch.atanh( 2. * flakes_transp_ - 1 ) # flakes_transp_ = backtransform to -infty ... infty
            # flakes_transp_ = (flakes_transp_ + 2.) / samples - 2.

            # Add new flake-3D positions in direction of motion blur
            offs = torch.tensor([i / (samples-1) * args.motionblur_scale]) # goes from (0 ... 1)*motionblur_scale
            offs = offs[:,None].repeat(1,4)
            offs = offs.to(points3D.device)
            motion_diff_ = motion_diff*offs[None,:]

            points3D_ = points3D + motion_diff_
            points3D_moved_ = points3D_moved + motion_diff_
        else:
            flakes_transp_ = flakes_transp
            points3D_ = points3D
            points3D_moved_ = points3D_moved

        image1_result = renderimg(image1_result, full_P, points3D_, flakes_, flakes_color_, flakes_transp_, depth1, args)
        image2_result = renderimg(image2_result, full_P, points3D_moved_, flakes_, flakes_color_, flakes_transp_, depth2, args)

    return image1_result, image2_result


def renderimg(image, full_P, points_3d, flakes, flakes_color, flakes_transp, image_depth, args):
    points_2d, flakes_depth = threeD_to_twoD(full_P, points_3d)
    points_2d_flipped = torch.flip(points_2d, [2])

    ### definitely differentiable, was tested separately
    image_result = twoD_to_img(image, image_depth, points_2d_flipped, flakes, flakes_color, flakes_transp, flakes_depth, args)
    image_result = torch.clamp(image_result,0,1)

    return image_result
