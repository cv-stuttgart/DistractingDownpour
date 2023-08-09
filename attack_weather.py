from __future__ import print_function
import mlflow
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os

from tqdm import tqdm
from mlflow import log_metric, log_param

from helper_functions import ownutilities, losses, parsing_file, targets, logging
from helper_functions.config_specs import Conf
from weather_attack.render import render
from weather_attack.weather import get_weather, recolor_weather
from weather_attack.utils import load_weather




def get_optimizer(optimizer_name, optimization_parameters, optimizer_lr=0.001):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(optimization_parameters, lr=optimizer_lr)
    else:
        raise ValueError("The selected optimizer option '%s' is unknown. Select 'Adam'." % (args.optimizer))
    return optimizer




def attack_image(model, image1, image2, flow, batch, distortion_folder, device, optimizer_lr, has_gt, scene_data, weather, args):

    # If the model takes unit input, ownutilities.preprocess_img will transform images into [0,1].
    # Otherwise, do transformation here
    if not ownutilities.model_takes_unit_input(args.net):
        image1 = image1/255.
        image2 = image2/255.

    eps_box = 1e-7
    full_P, ext1, rel_mat, gt1_depth, gt2_depth = scene_data

    # RAFT input padding
    padder, [image1, image2] = ownutilities.preprocess_img(args.net, image1, image2)
    gt1_depth = F.pad(gt1_depth, padder.get_pad_dims(), "replicate")
    gt2_depth = F.pad(gt2_depth, padder.get_pad_dims(), "replicate")

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False
    gt1_depth.requires_grad = False
    gt2_depth.requires_grad = False
    full_P.requires_grad = False
    ext1.requires_grad = False
    rel_mat.requires_grad = False

    scene_data = full_P, ext1, rel_mat, gt1_depth, gt2_depth
    initpos, motion, flakes, flakes_color, flakes_transp = weather

    print("Saving initial weather")
    if batch % args.save_frequency == 0 and not args.no_save:
        image1_weather_init, image2_weather_init = render(image1.detach().clone(), image2.detach().clone(), scene_data, weather, args)
        [image1_weather_init, image2_weather_init] = ownutilities.postprocess_flow(args.net, padder, image1_weather_init, image2_weather_init)
        logging.save_image(image1_weather_init, batch, distortion_folder, image_name='img1_init', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(image2_weather_init, batch, distortion_folder, image_name='img2_init', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)


    # ===== ATTACK OPTIMIZATION =====

    torch.autograd.set_detect_anomaly(True)
    curr_step = batch*args.steps

    # prepare and rescale images
    offsets = torch.zeros_like(initpos).detach().clone()
    motion_offsets = torch.zeros_like(initpos)

    flow_weather_init = None
    flakes_transp_init = flakes_transp.clone().detach()
    flakes_color_init = flakes_color.clone().detach()
    offset_init = offsets.detach().clone()

    # non-learnable parameters
    initpos.requires_grad = False
    flakes.requires_grad = False
    motion.requires_grad = False
    flakes_color.requires_grad = False
    flakes_color_inf = torch.atanh( 2. * (1.- eps_box) * (flakes_color) - (1 - eps_box)  ) # switch color to [-infty, infty] for better optimization
    # Note: the transparencies are already atanh-transformed

    # learnable parameters
    offsets.requires_grad = True
    motion_offsets.requires_grad = True
    flakes_transp.requires_grad = True
    flakes_color_inf.requires_grad = True

    learnparams = []
    if args.learn_offset:
        learnparams += [offsets]
    if args.learn_motionoffset:
        learnparams += [motion_offsets]
    if args.learn_transparency:
        learnparams += [flakes_transp]
    if args.learn_color:
        learnparams += [flakes_color_inf]
    if learnparams == []:
        raise ValueError("No learnable parameters were passed in the argument parser. Cannot optimize particles.")
    optimizer = get_optimizer(args.optimizer, learnparams, optimizer_lr=optimizer_lr)

    # Predict the flow
    flakes_color_img = (1./2.) * 1. / (1. - eps_box) * (torch.tanh(flakes_color_inf) + (1 - eps_box) )
    weather = (initpos+offsets, motion+motion_offsets, flakes, flakes_color_img, flakes_transp)
    flow_weather = ownutilities.compute_flow(model, "scaled_input_weather_model", image1, image2, test_mode=True, weather=weather, scene_data=scene_data, args_=args)
    [flow_weather] = ownutilities.postprocess_flow(args.net, padder, flow_weather)
    flow_weather = flow_weather.to(device)

    # define the initial flow, the target, and update mu
    flow_weather_init = flow_weather.detach().clone()
    flow_weather_init.requires_grad = False

    flow_init = ownutilities.compute_flow(model, "scaled_input_weather_model", image1, image2, test_mode=True)
    [flow_init] = ownutilities.postprocess_flow(args.net, padder, flow_init)
    flow_init = flow_init.to(device).detach()

    if batch % args.save_frequency == 0 and not args.no_save:
        logging.save_tensor(flow_init, "flow_init", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_weather_init, "flow_weather_init", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)

    # define target (potentially based on first flow prediction)
    # define attack target
    target = targets.get_target(args.target, flow_init, device=device)
    target = target.to(device)
    target.requires_grad = False

    # initialize values and best values
    # EPE statistics for the unattacked flow
    aee_gt = 0.
    aee_gt_tgt = 0.
    aee_adv_gt = 0.
    aee_tgt            = logging.calc_metrics_const(target, flow_init)
    aee_gt_tgt, aee_gt = logging.calc_metrics_const_gt(target, flow_init, flow) if has_gt else (None, None)
    logging.log_metrics(batch*args.steps, ("aee_prd-tgt", aee_tgt),
                                    ("aee_grt-tgt", aee_gt_tgt),
                                    ("aee_prd-grt", aee_gt))
    # And for the initial weather prediction
    aee_adv_tgt_init, aee_adv_pred_init = logging.calc_metrics_adv(flow_weather_init, target, flow_init)
    aee_adv_gt_init                = logging.calc_metrics_adv_gt(flow_weather_init, flow) if has_gt else None
    logging.log_metrics(batch*args.steps, ("aee_adv-tgt_init", aee_adv_tgt_init),
                                    ("aee_prd-adv_init", aee_adv_pred_init),
                                    ("aee_adv-grt_init", aee_adv_gt_init))
    aee_adv_tgt_min_val = aee_adv_tgt_init
    aee_adv_pred_min_val = aee_adv_pred_init
    aee_adv_gt_min_val = aee_adv_gt_init

    flow_weather_min = flow_weather_init.clone().detach()
    offsets_min = offsets.clone().detach()
    motion_offsets_min = motion_offsets.clone().detach()
    flakes_color_min = flakes_color_img.clone().detach()
    flakes_transp_min = flakes_transp.clone().detach()

    # Zero all existing gradients
    model.zero_grad()
    optimizer.zero_grad()

    for steps in range(args.steps):

        curr_step = batch*args.steps + steps
        log_metric(key="batch", value=batch, step=curr_step)
        log_metric(key="steps", value=steps, step=curr_step)
        log_metric(key="epoch", value=0, step=curr_step)

        # Calculate loss
        loss = losses.loss_weather(flow_weather, target, f_type=args.loss, init_pos=initpos, offsets=offsets, motion_offsets=motion_offsets, flakes_transp=flakes_transp, flakes_transp_init=flakes_transp_init, alph_offsets=args.alph_motion, alph_motion=args.alph_motionoffset, alph_transp=0)
 
        loss.backward()

        if args.optimizer in ['Adam']:
            optimizer.step()
        else:
            raise RuntimeWarning('Unknown optimizer, no optimization step was performed')

        flakes_color_img = (1./2.) * 1. / (1. - eps_box) * (torch.tanh(flakes_color_inf) + (1 - eps_box) )
        weather = (initpos+offsets, motion+motion_offsets, flakes, flakes_color_img, flakes_transp)

        flow_weather = ownutilities.compute_flow(model, "scaled_input_weather_model", image1, image2, weather=weather, scene_data=scene_data, test_mode=True, args_=args)
        [flow_weather] = ownutilities.postprocess_flow(args.net, padder, flow_weather)
        flow_weather = flow_weather.to(device)

        # More AEE statistics, now for attacked images
        aee_adv_tgt, aee_adv_pred = logging.calc_metrics_adv(flow_weather, target, flow_weather_init)
        aee_adv_gt                = logging.calc_metrics_adv_gt(flow_weather, flow) if has_gt else None
        logging.log_metrics(curr_step, ("aee_adv-tgt", aee_adv_tgt),
                                        ("aee_prd-adv", aee_adv_pred),
                                        ("aee_adv-grt", aee_adv_gt))

        if aee_adv_tgt < aee_adv_tgt_min_val:
            aee_adv_tgt_min_val = aee_adv_tgt
            aee_adv_pred_min_val = aee_adv_pred
            aee_adv_gt_min_val = aee_adv_gt
            flow_weather_min = flow_weather.clone().detach()
            offsets_min = offsets.clone().detach()
            motion_offsets_min = motion_offsets.clone().detach()
            flakes_color_min = flakes_color_img.clone().detach()
            flakes_transp_min = flakes_transp.clone().detach()

        logging.log_metrics(curr_step, ("aee_adv-tgt_min", aee_adv_tgt_min_val),
                                        ("aee_prd-adv_min", aee_adv_pred_min_val))

    if batch % args.save_frequency == 0 and not args.no_save:

        # only rendered initial weather images saved
        logging.save_tensor(flow_weather_min, "flow_weather_best", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_weather_init, "flow_weather_init", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_init, "flow_init", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(target, "targ", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)

        weather_min = (initpos+offsets_min, motion+motion_offsets_min, flakes, flakes_color_min, flakes_transp_min)
        image1_weather, image2_weather = render(image1.detach().clone(), image2.detach().clone(), scene_data, weather_min, args)
        [image1_weather, image2_weather] = ownutilities.postprocess_flow(args.net, padder, image1_weather, image2_weather)
        logging.save_image(image1_weather, batch, distortion_folder, image_name='img1_best', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(image2_weather, batch, distortion_folder, image_name='img2_best', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)

        if not args.no_flake_dat:
            weather_min = [i.to("cpu") for i in weather_min]
            logging.save_weather(weather_min, batch, distortion_folder, weather_name="weath_best", unregistered_artifacts=args.unregistered_artifacts)

        max_flow_gt = 0
        if has_gt:
            max_flow_gt = ownutilities.maximum_flow(flow)
        max_flow = np.max([max_flow_gt,
                            ownutilities.maximum_flow(flow_weather_init),
                            ownutilities.maximum_flow(flow_weather_min),
                            ownutilities.maximum_flow(target),
                            ownutilities.maximum_flow(flow_init)])

        logging.save_flow(flow_weather_min, batch, distortion_folder, flow_name='flow_weather_best', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_flow(target, batch, distortion_folder, flow_name='flow_targ', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_flow(flow_weather_init, batch, distortion_folder, flow_name='flow_weather_init', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_flow(flow_init, batch, distortion_folder, flow_name='flow_init', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)

    return aee_gt, aee_tgt, aee_gt_tgt, aee_adv_gt_init, aee_adv_tgt_init, aee_adv_pred_init, aee_adv_gt_min_val, aee_adv_tgt_min_val, aee_adv_pred_min_val



def attack_dataset(args):
    """
    Performs a weather attack on a given model and for all images of a specified dataset.
    """

    experiment_id, folder_path, folder_name = logging.mlflow_experimental_setup(args.output_folder, args.net, "Weather", False, False)

    optimizer_lr = args.lr
    if args.lr == 0. and args.delta_bound > 0.:
        optimizer_lr = args.delta_bound
    elif args.lr == 0. and  args.delta_bound == 0.:
        raise ValueError("No optimizer learning rate was specified, and neither was a delta_bound given. It is unclear which learning rate should be used. Please specify one using the --lr argument when calling the attack. Aborting.")

    print("\nStarting Weather Attack (Weather):")
    print()
    print("\tModel:                      %s" % (args.net))
    print("\tFrames per scene:           %d" % (args.frame_per_scene))
    print("\tScenes scale:               %f" % (args.scene_scale))
    print()
    print("\tTarget:                     %s" % (args.target))
    print("\tOptimizer:                  %s" % (args.optimizer))
    print("\tOptimizer steps:            %d" % (args.steps))
    print("\tOptimizer boxconstraint:    %s" % ('clipping'))
    print("\tOptimizer LR:               %f" % (optimizer_lr))
    print()
    print("\tNumber of flakes:           %d" % (args.num_flakes))
    print("\tMaximal size flakes:        %f" % (args.flakesize_max))
    print("\tFlake motion x:             %f" % (args.motion_x))
    print("\tFlake motion y:             %f" % (args.motion_y))
    print("\tFlake motion z:             %f" % (args.motion_z))
    print("\tTransparency scale:         %f" % (args.transparency_scale))
    print("\tLearnable initpos:          %s" % (args.learn_offset))
    print("\tLearnable motion offset:    %s" % (args.learn_motionoffset))
    print("\tLearnable transparency:     %s" % (args.learn_transparency))
    print("\tLearnable color:            %s" % (args.learn_color))
    print("\tDepth check:                %s" % (args.depth_check))
    print("\tDepth check differentiable: %s" % (args.depth_check_differentiable))
    print()
    print("\tFlakes deterministic:       %s" % (args.deterministic_startpos))
    print("\tFlakes rotate:              %s" % (args.do_rotate))
    print("\tFlakes scale:               %s" % (args.do_scale))
    print("\tFlakes blur:                %s" % (args.do_blur))
    print("\tFlakes motionblur:          %s" % (args.do_motionblur))
    print("\tFlakes transp:              %s" % (args.do_transp))
    print("\tFlakes shape:               %s" % (args.do_shape))
    print()
    print("\tLoss weight motion:         %s" % (args.alph_motion))
    print("\tLoss weight motion offset:  %s" % (args.alph_motionoffset))
    print()
    print("\tOutputfolder:               %s" % (folder_path))
    print()

    with mlflow.start_run(experiment_id=experiment_id, run_name=folder_name):

        log_param("outputfolder", folder_path)
        distortion_folder_name = "patches"
        distortion_folder_path = folder_path
        distortion_folder = logging.create_subfolder(distortion_folder_path, distortion_folder_name)

        model_takes_unit_input = ownutilities.model_takes_unit_input(args.net)
        logging.log_model_params(args.net, model_takes_unit_input)
        logging.log_dataset_params(args, 1, 1, False)
        logging.log_attack_params("Weather", args.loss, args.target, False, False)
        log_param("optimizer", args.optimizer)
        log_param("optimizer_boxconstraint", "clipping")
        log_param("optimizer_steps", args.steps)
        log_param("optimizer_lr", optimizer_lr)
        log_param("flakes_number", args.num_flakes)
        log_param("flakes_maxsize", args.flakesize_max)
        log_param("flakes_motion_x", args.motion_x)
        log_param("flakes_motion_y", args.motion_y)
        log_param("flakes_motion_z", args.motion_z)
        log_param("flakes_transparency_scale", args.transparency_scale)
        log_param("flakes_learn_offset", args.learn_offset)
        log_param("flakes_learn_motionoffset", args.learn_motionoffset)
        log_param("flakes_learn_transparency", args.learn_transparency)
        log_param("flakes_learn_color", args.learn_color)
        log_param("flakes_depthcheck", args.depth_check)
        log_param("flakes_depthcheck_differentiable", args.depth_check_differentiable)
        log_param("flakes_no_data", not args.no_flake_dat)
        log_param("flakes_start_determinist", args.deterministic_startpos)
        log_param("flakes_do_rotate", args.do_rotate)
        log_param("flakes_do_scale", args.do_scale)
        log_param("flakes_do_blur", args.do_blur)
        log_param("flakes_do_motionblur", args.do_motionblur)
        log_param("flakes_do_transp", args.do_transp)
        log_param("flakes_do_shape", args.do_shape)
        log_param("loss_alph_motion", args.alph_motion)
        log_param("loss_alph_motionoffset", args.alph_motionoffset)
        log_param("weather_data", args.weather_data)

        print("Preparing data from %s %s" % (args.dataset, args.dataset_stage))
        print(f"Loading weather data from {args.weather_data}.")
        data_loader, has_gt, has_cam, has_weather = ownutilities.prepare_dataloader(args, shuffle=False, get_weather=True)

        if not has_cam:
            print("The datset '%s' at stage '%s' does not contain information about the camera data, which is necessary for a weather attack. Please use a dataset that provides camera data.\nAborting." % (args.dataset, args.dataset_stage))
            exit()

        # Define what device we are using
        if Conf.config('useCPU') or not torch.cuda.is_available() or args.cpu_only:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        print("Setting Device to %s\n" % device)

        # Initialize the network
        # load model that is configured for training particles, which takes images scaled to [0,1] as input
        print("Loading model %s:" % (args.net))
        model, path_weights = ownutilities.import_and_load(args.net, custom_weight_path=args.custom_weight_path, make_unit_input=not model_takes_unit_input, variable_change=False, make_scaled_input_weather_model=True, device=device)
        log_param("model_path_weights", path_weights)

        # Set the model in evaluation mode. This can be needed for Dropout layers, and is also required for the BatchNorm2dLayers in RAFT (that would otherwise still change in training)
        model.eval()
        # Make sure the model is not trained:
        for param in model.parameters():
            param.requires_grad = False


        # Initialize statistics and Logging
        sum_aee_gt = 0.
        sum_aee_tgt = 0.
        sum_aee_gt_tgt = 0.
        sum_aee_adv_gt_init = 0.
        sum_aee_adv_tgt_init = 0.
        sum_aee_adv_pred_init = 0.
        sum_aee_adv_gt_min = 0.
        sum_aee_adv_tgt_min = 0.
        sum_aee_adv_pred_min = 0.
        sum_aee_adv_gt_min_val = 0.
        tests = 0


        # Loop over all examples in test set
        print("Starting Weather Attack on %s %s\n" % (args.dataset, args.dataset_stage))
        for batch, datachunck in enumerate(tqdm(data_loader)):

            if has_weather:
              (image1, image2, image1_weather, image2_weather, flow, _, scene_data, extra) = datachunck
            else:
              raise ValueError("Cannot evaluate weather without weather data. Please pass --weather_data to the argument parser.")

            (root,), (split,), (seq,), (base,), (frame,), (weatherdat,) = extra
            weather = get_weather(has_weather, weatherdat, scene_data, args, seed=None, load_only=True)


            scene_data = [i.to(device) for i in scene_data]
            weather = [i.to(device) for i in weather]
            image1, image2 = image1.to(device), image2.to(device)
            flow = flow.to(device)

            aee_gt, aee_tgt, aee_gt_tgt, aee_adv_gt_init, aee_adv_tgt_init, aee_adv_pred_init, aee_adv_gt_min_val, aee_adv_tgt_min_val, aee_adv_pred_min_val = attack_image(model, image1, image2, flow, batch, distortion_folder, device, optimizer_lr, has_gt, scene_data, weather, args)
            sum_aee_tgt += aee_tgt
            sum_aee_adv_tgt_init += aee_adv_tgt_init
            sum_aee_adv_pred_init += aee_adv_pred_init
            sum_aee_adv_tgt_min += aee_adv_tgt_min_val
            sum_aee_adv_pred_min += aee_adv_pred_min_val
            if has_gt:
                sum_aee_gt += aee_gt
                sum_aee_gt_tgt += aee_gt_tgt
                sum_aee_adv_gt_init += aee_adv_gt_init
                sum_aee_adv_gt_min_val += aee_adv_gt_min_val
            tests += 1

        # Calculate final accuracy
        logging.calc_log_averages(tests,
                ("aee_avg_prd-grt",sum_aee_gt),
                ("aee_avg_prd-tgt", sum_aee_tgt),
                ("aee_avg_grt-tgt",sum_aee_gt_tgt),
                ("aee_avg_adv-grt_init", sum_aee_adv_gt_init),
                ("aee_avg_adv-tgt_init", sum_aee_adv_tgt_init),
                ("aee_avg_prd-adv_init", sum_aee_adv_pred_init),
                ("aee_avg_adv-grt_min", sum_aee_adv_gt_min_val),
                ("aee_avg_adv-tgt_min", sum_aee_adv_tgt_min),
                ("aee_avg_prd-adv_min", sum_aee_adv_pred_min)
                )


        print("\nFinished attacking with weather. The best achieved values are")
        print("\tAEE(f_adv, f_init)=%f" % (sum_aee_adv_pred_min / tests))
        print("\tAEE(f_adv, f_targ)=%f" % (sum_aee_adv_tgt_min / tests))
        print()

if __name__ == '__main__':

    parser = parsing_file.create_parser(stage='training', attack_type='weather')
    args = parser.parse_args()
    print(args)

    attack_dataset(args)
