from __future__ import print_function
import mlflow
import torch
import re
import os
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from mlflow import log_param

from helper_functions import ownutilities,parsing_file, targets, logging
from helper_functions.config_specs import Conf
from weather_attack import weather, render

from generate_weather import toggle_vis



def get_flake_data(batch, path, device):

    flks = None
    posn_init = None
    motn = None
    offs_best = None
    omot_best = None
    trsp_best = None

    if os.path.exists(path):
        print("Enter folder")
        number = "{:05d}".format(batch)
        pattern_flks = re.compile(number + "_flks.npy")
        pattern_posn_init = re.compile(number + "_posn_init.npy")
        pattern_motn = re.compile(number + "_motn.npy")
        pattern_offs_best = re.compile(number + "_offs_best.npy")
        pattern_omot_best = re.compile(number + "_omot_best.npy")
        pattern_trsp_best = re.compile(number + "_trsp_best.npy")

        base_folder = os.path.join(path, "patches")

        for file in os.listdir(base_folder):
            if pattern_flks.match(file):
                flks = np.load(os.path.join(base_folder,file))
            if pattern_posn_init.match(file):
                posn_init = np.load(os.path.join(base_folder,file))
            if pattern_motn.match(file):
                motn = np.load(os.path.join(base_folder,file))
            if pattern_offs_best.match(file):
                offs_best = np.load(os.path.join(base_folder,file))
            if pattern_omot_best.match(file):
                omot_best = np.load(os.path.join(base_folder,file))
            if pattern_trsp_best.match(file):
                trsp_best = np.load(os.path.join(base_folder,file))

    return torch.tensor(flks).to(device), torch.tensor(posn_init).to(device), torch.tensor(motn).to(device), torch.tensor(offs_best).to(device), torch.tensor(omot_best).to(device), torch.tensor(trsp_best).to(device)



def evaluate_weather(args):
    """
    Performs an weather evaluation on a given model and for all images of a specified dataset.
    """

    experiment_id, folder_path, folder_name = logging.mlflow_experimental_setup(args.output_folder, args.net, "Weather", False, False, stage="eval")

    # optimizer_lr = args.lr
    # if args.lr == 0. and args.delta_bound > 0.:
    #     optimizer_lr = args.delta_bound
    # elif args.lr == 0. and  args.delta_bound == 0.:
    #     raise ValueError("No optimizer learning rate was specified, and neither was a delta_bound given. It is unclear which learning rate should be used. Please specify one using the --lr argument when calling the attack. Aborting.")

    print("\nEvaluating Weather Augmentation (Weather):")
    print()
    print("\tModel (evaluation, now):    %s" % (args.net))
    print("\tFrames per scene:           %d" % (args.frame_per_scene))
    print("\tScenes scale:               %f" % (args.scene_scale))
    print("\tSingle Scene:               %s" % (args.single_scene))
    print("\tFrom Scene:                 %s" % (args.from_scene))
    print()
    print("\tTarget:                     %s" % (args.target))
    print("\tDepth check:                %s" % (args.depth_check))
    print()
    print("\tWeather data:               %s" % (args.weather_data))
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
        logging.log_attack_params("Weather", None, args.target, False, False)
        log_param("weather_data", args.weather_data)
        log_param("flakes_depthcheck", args.depth_check)

        print("Preparing data from %s %s\n" % (args.dataset, args.dataset_stage))
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
        sum_aee_adv_gt = 0.
        sum_aee_adv_tgt = 0.
        sum_aee_adv_pred = 0.
        sum_aee_adv_tgt_min = 0.
        sum_aee_adv_pred_min = 0.
        tests = 0

        # Loop over all examples in test set
        print("Starting Evaluation on %s %s\n" % (args.dataset, args.dataset_stage))
        for batch, datachunck in enumerate(tqdm(data_loader)):

            if has_weather:
              (image1, image2, image1_weather, image2_weather, flow, _, scene_data, extra) = datachunck
            else:
              raise ValueError("Cannot evaluate weather without weather data. Please pass --weather_data to the argument parser.")

            image1, image2 = image1.to(device), image2.to(device)
            image1_weather, image2_weather = image1_weather.to(device), image2_weather.to(device)
            flow = flow.to(device)

            not_rendered = torch.equal(image1_weather[0,:,:50,:50], torch.zeros_like(image1_weather[0,:,:50,:50]))
            if not_rendered:
                (root,), (split,), (seq,), (base,), (frame,), (weatherdat,) = extra
                raise ValueError("Images not rendered. This should not happen for this test case.")
                weather = weather.get_weather(has_weather, weatherdat, scene_data, args, seed=None, load_only=True)

                scene_data = [i.to(device) for i in scene_data]
                weather = [i.to(device) for i in weather]
                for w in weather:
                    w.requires_grad = False

                image1_weather, image2_weather = render.render(image1/255, image2/255, scene_data, weather, args)
                image1_weather *= 255.
                image2_weather *= 255
                print(torch.max(image1_weather))

            # If the model takes unit input, ownutilities.preprocess_img will transform images into [0,1].
            # Otherwise, do transformation here
            if not ownutilities.model_takes_unit_input(args.net):
                image1 = image1/255.
                image2 = image2/255.
                image1_weather = image1_weather/255.
                image2_weather = image2_weather/255.

            padder, [image1, image2] = ownutilities.preprocess_img(args.net, image1, image2)
            padder, [image1_weather, image2_weather] = ownutilities.preprocess_img(args.net, image1_weather, image2_weather)

            flow_pred_init = ownutilities.compute_flow(model, "scaled_input_weather_model", image1, image2, test_mode=True)
            [flow_pred_init] = ownutilities.postprocess_flow(args.net, padder, flow_pred_init)
            flow_pred_init = flow_pred_init.to(device)

            flow_pred_weather = ownutilities.compute_flow(model, "scaled_input_weather_model", image1_weather, image2_weather, test_mode=True)
            [flow_pred_weather] = ownutilities.postprocess_flow(args.net, padder, flow_pred_weather)
            flow_pred_weather = flow_pred_weather.to(device)

            target = targets.get_target(args.target, flow_pred_init.clone().detach(), device=device)
            target = target.to(device)
            target.requires_grad = False


            # Some EPE statistics for the unattacked flow
            aee_tgt            = logging.calc_metrics_const(target, flow_pred_init)
            aee_gt_tgt, aee_gt = logging.calc_metrics_const_gt(target, flow_pred_init, flow) if has_gt else (None, None)
            logging.log_metrics(batch, ("aee_prd-tgt", aee_tgt),
                                           ("aee_grt-tgt", aee_gt_tgt),
                                           ("aee_prd-grt", aee_gt))

            # AEE statistics, for attacked images
            aee_adv_tgt, aee_adv_pred = logging.calc_metrics_adv(flow_pred_weather, target, flow_pred_init)
            aee_adv_gt                = logging.calc_metrics_adv_gt(flow_pred_weather, flow) if has_gt else None
            logging.log_metrics(batch, ("aee_adv-tgt", aee_adv_tgt),
                                       ("aee_prd-adv", aee_adv_pred),
                                       ("aee_adv-grt", aee_adv_gt))

            sum_aee_tgt += aee_tgt
            sum_aee_adv_tgt += aee_adv_tgt
            sum_aee_adv_pred += aee_adv_pred
            sum_aee_adv_tgt_min += aee_adv_tgt
            sum_aee_adv_pred_min += aee_adv_pred
            if has_gt:
                sum_aee_gt += aee_gt
                sum_aee_gt_tgt += aee_gt_tgt
                sum_aee_adv_gt += aee_adv_gt
            tests += 1

            if batch % args.save_frequency == 0 and not args.no_save:
                logging.save_tensor(flow_pred_init, "flow_best", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(flow_pred_weather, "flow_init", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                max_flow_gt = 0
                if has_gt:
                    max_flow_gt = ownutilities.maximum_flow(flow)
                max_flow = np.max([max_flow_gt,
                                   ownutilities.maximum_flow(flow_pred_weather),
                                   ownutilities.maximum_flow(flow_pred_init)])
                logging.save_flow(flow_pred_weather, batch, distortion_folder, flow_name='flow_best', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)

        # Calculate final accuracy
        logging.calc_log_averages(tests,
                ("aee_avg_prd-grt",sum_aee_gt),
                ("aee_avg_prd-tgt", sum_aee_tgt),
                ("aee_avg_grt-tgt",sum_aee_gt_tgt),
                ("aee_avg_adv-grt", sum_aee_adv_gt),
                ("aee_avg_adv-tgt", sum_aee_adv_tgt),
                ("aee_avg_prd-adv", sum_aee_adv_pred),
                ("aee_avg_adv-tgt_min", sum_aee_adv_tgt_min),
                ("aee_avg_prd-adv_min", sum_aee_adv_pred_min)
                )

        print("\nFinished evaluating the weather. The best achieved values are")
        print("\tAEE(f_adv, f_init)=%f" % (sum_aee_adv_pred_min / tests))
        print("\tAEE(f_adv, f_targ)=%f" % (sum_aee_adv_tgt_min / tests))
        print()

if __name__ == '__main__':

    parser = parsing_file.create_parser(stage='evaluation', attack_type='weather')
    args = parser.parse_args()
    print(args)

    evaluate_weather(args)
