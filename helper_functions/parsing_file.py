import argparse
import json
import os
import ast


def create_parser(stage=None, attack_type=None):
    stage = stage.lower()
    attack_type = attack_type.lower()
    if stage not in ['training', 'evaluation', 'generation']:
        raise ValueError('To create a parser the stage has to be specified. Please choose one of "training", "evaluation", or "generation"')


    parser = argparse.ArgumentParser(usage='%(prog)s [options (see below)]')

    # network arguments
    glob_args = parser.add_argument_group(title='global arguments')
    glob_args.add_argument('--cpu_only', default=False, type=ast.literal_eval,
        help="if toggled, this ensures CPU only execution (default=False).")

    # network arguments
    net_args = parser.add_argument_group(title='network arguments')
    net_args.add_argument('--net', default='SpyNet', choices=['RAFT', 'GMA', 'FlowFormer', 'SpyNet', 'FlowNet2', 'FlowNetCRobust'],
        help="specify the network under attack")
    net_args.add_argument('--custom_weight_path', default='',help="specify path to load weights from. By default loads to the given net default weights from `models/_pretrained_weights`")

    # Dataset arguments
    dataset_args = parser.add_argument_group(title="dataset arguments")
    dataset_args.add_argument('--dataset', default='Sintel',
        help="specify the dataset which should be used for evaluation")
    dataset_args.add_argument('--dataset_stage', default='training', choices=['training', 'evaluation'],
        help="specify the dataset stage ('training' or 'evaluation') that should be used.")
    dataset_args.add_argument('--small_run', action='store_true',
        help="for testing purposes: if specified the dataloader will on load 32 images")
    # Sintel specific:
    sintel_args = parser.add_argument_group(title="sintel specific arguments")
    sintel_args.add_argument('--dstype', default='final', choices=['clean', 'final'],
        help="[only sintel] specify the dataset type for the sintel dataset")
    sintel_args.add_argument('--frame_per_scene', default=0, type=int,
        help="the number of optimization scenes per sintel-sequence (if 0, all scenes per sequence are taken).")
    sintel_args.add_argument('--single_scene', default='',
        help='if a scene name is added, only this sintel-scene is in the dataset.')
    sintel_args.add_argument('--from_scene', default='',
        help='if a scene is specified as from_scene, then all subsequent scenes (alphabetic order, including the specified scene) are added to the dataset.')
    sintel_args.add_argument('--from_frame', default=0, type=int,
        help='For all scenes, the dataset starts from specified frame number. 0 takes all frames, 1 all except first, etc.')
    sintel_args.add_argument('--scene_scale', default=1.0, type=float,
        help="A global scaling to the scene depth. If the value is > 1, all scenes will appear bigger and more particles will show up in the foreground.")

    # Data saving
    data_save_args = parser.add_argument_group(title="data saving arguments")
    data_save_args.add_argument('--output_folder', default='experiment_data',
        help="data that is logged during training and evaluation will be saved there")
    data_save_args.add_argument('--save_frequency', type=int, default=1,
            help="specifies after how many batches intermediate results (patch, input images, flows) should be saved. Default: 1 (save after every batch/image). If --no_save is specified, this overwrites any save_frequency.")
    data_save_args.add_argument('--no_save', action='store_true',
        help="if specified no extended output (like distortions/patches) will be written. This overwrites any value specified by save_frequency.")
    data_save_args.add_argument('--unregistered_artifacts', default=True, type=ast.literal_eval,
        help="if True, artifacts are saved to the output folder but not registered. Saves time and memory during training.")
    data_save_args.add_argument('--custom_experiment_name', default='',
        help="specify a custom mlflow experiment name. The given string is concatenated with the automatically generated one.")

    # Target setup
    target_args = parser.add_argument_group(title="target flow arguments")
    target_args.add_argument('--target', default='zero', choices=['zero'],
        help="specify the attack target.")

    # Arguments for training:
    if stage == 'training':
        train_args = parser.add_argument_group(title="training arguments")
        train_args.add_argument('--loss', default='aee', choices=['aee', 'mse', 'cosim'],
            help="specify the loss function as one of 'aee', 'cosim' or 'mse'")


    # Weather unique arguments
    if attack_type in ['weather']:
        data_save_args.add_argument('--no_flake_dat', default=True, type=ast.literal_eval,
            help="if this flag is used, no data about the particle (positions, flakes, transparencies) will be stored.")

        weather_args = parser.add_argument_group(title="weather arguments")

        weather_args.add_argument('--weather_data', default='',
                help="may specify a dataset that contains weather data (locations, masks, etc). It should have the same structure as the used dataset.")

        # Universal rendering arguments (also needed if flake data given)
        weather_args.add_argument('--rendering_method', default='additive', choices=['meshkin', 'additive'],
            help="choose a method rendering the particle color. 'meshkin' use alpha-blending with order-independent transparency calculation, while 'additive' adds the color value to the image. Default: 'meshkin', choices: [meshkin, additive].")
        weather_args.add_argument('--transparency_scale', default=1., type=float,
            help="a scaling factor, by which the tansparency for every particle is multiplied.")
        weather_args.add_argument('--depth_check', default=False, type=ast.literal_eval,
            help="if specified, particles will not be rendered if behind an object.")

        # Motionblur arguments
        weather_args.add_argument('--do_motionblur', default=True, type=ast.literal_eval,
            help="control if particles are rendered with motion blur (default=True).")
        weather_args.add_argument('--motionblur_scale', default=.025, type=float,
            help="a scaling factor in [0,1], by which the motion blur is shortened. No motion blur appears for 0, while the full blur vector is used with 1. A full motion blur might need a higher number of motionblur_samples.")
        weather_args.add_argument('--motionblur_samples', default=10, type=int,
            help="the number of flakes that is drawn per blurred flake. More samples are needed for faster objects or a larger motionblur_scale.")

        weather_args.add_argument('--recolor', default=False, type=ast.literal_eval,
            help="If specified, all weather is recolored with the given r,g,b value (no variations).")
        weather_args.add_argument('--flake_r', default=255, type=int,
            help="the R value for the particle RGB")
        weather_args.add_argument('--flake_g', default=255, type=int,
            help="the G value for the particle RGB")
        weather_args.add_argument('--flake_b', default=255, type=int,
            help="the B value for the particle RGB")


        if stage == "generation":
            weather_args.add_argument('--save_weather', default=True, type=ast.literal_eval,
                help="control if generated weather data is saved (default=True). Otherwise it is visualized")
            weather_args.add_argument('--save_images', default=True, type=ast.literal_eval,
                help="control if the rendered images are saved (default=False). Otherwise it is visualized.")
            weather_args.add_argument('--weather_path', default='',
                help="specify a path where a new folder for the saved weather will be created.")
            weather_args.add_argument('--weather_tag', default='',
                help="specify custom weather description, to append to the dataset name when the weather is saved.")

        if stage == "training" or stage == "generation":
            weather_args.add_argument('--cpu_count', default=0, type=int,
                help="The number of cpus for parallel particle generation. If set to 0, half of the available GPUS will be used.")

            weather_args.add_argument('--num_flakes', default=1000, type=int,
                help="the number of particles that will be generated initially.")
            weather_args.add_argument('--flakesize_max', default=71, type=int,
                help="the maximal size for particles in pixels.")
            weather_args.add_argument('--flake_folder', default=os.path.join("weather_attack","billboards"),
                help="the folder from which the flake templates are loaded")
            weather_args.add_argument('--flake_template_folder', default="particles",
                help="the folder within flake_folder, from where the flake templates are loaded. Useful to differentiate between particles / dust billboards.")
            weather_args.add_argument('--flake_random_h', default=0, type=float,
                help="the upper bound for HSL color Hue (H) randomization. Hue runs from 0° to 360°, hence values >= 180 will give fully randomized hues.")
            weather_args.add_argument('--flake_random_s', default=0, type=float,
                help="the upper bound for HSL color Saturation (S) randomization. Saturations runs from 0 (unsaturated) through 1 (fully saturated).")
            weather_args.add_argument('--flake_random_l', default=0, type=float,
                help="the upper bound for HSL color Lightness (L) randomization. Lightness runs from 0 (black) over 0.5 (color) to 1 (white).")

            weather_args.add_argument('--depth_check_differentiable', default=False, type=ast.literal_eval,
                help="if specified, the rendering check for particle occlusion by objects is included into the compute graph.")
            weather_args.add_argument('--depth_decay', default=10, type=float,
                help="a decay factor for the particle template size by depth. The particle template size is 1/depth/depth_decay.")
            weather_args.add_argument('--constant_transparency', default=0, type=float,
                help="if set to a value != 0, this is the default transparency for all initialized particles. Otherwise, the transparency is a hat-function that reaches its peak at a depth of 2.")

            weather_args.add_argument('--motion_random_scale', default=0.0, type=float,
                help="randomizes the magnitude of the particle motion relative to the motion vector length. By setting to 0.5, the motion vector can be longer or smaller up to half its length. (default=0.0)")
            weather_args.add_argument('--motion_random_angle', default=0.0, type=float,
                help="maximal random offset angle for the particle motion in degree. (default=0.0, max=180)")

            weather_args.add_argument('--motion_x', default=0., type=float,
                help="the motion in x-direction for all particles between frames.")
            weather_args.add_argument('--motion_y', default=0., type=float,
                help="the motion in y-direction for all particles between frames.")
            weather_args.add_argument('--motion_z', default=0., type=float,
                help="the motion in z-direction (depth) for all particles between frames.")
            weather_args.add_argument('--deterministic_startpos', default=False, type=ast.literal_eval,
                help="if True specified, particles are initialized deterministically (for reproducability).")
            weather_args.add_argument('--deterministic_startpos_seed', default=0, type=int,
                help="if particles are initialized deterministically (--deterministic_startpos), this value is added to the seed.")

            weather_args.add_argument('--do_rotate', default=True, type=ast.literal_eval,
                help="control if particles are rotated during initialization (default=True).")
            weather_args.add_argument('--do_scale', default=True, type=ast.literal_eval,
                help="control if particles are scaled during initialization (default=True).")
            weather_args.add_argument('--do_blur', default=True, type=ast.literal_eval,
                help="control if particles are blurred during initialization (default=True).")
            weather_args.add_argument('--do_transp', default=True, type=ast.literal_eval,
                help="control if particles are made transparent during initialization (default=True).")
            weather_args.add_argument('--do_shape', default=True, type=ast.literal_eval,
                help="control if particles are shaped (default=True). If False they are  kept as boxes during initialization.")

        if stage == "training":
            weather_args.add_argument('--steps', default=750, type=int,
                help="the number of optimization steps per image.")
            weather_args.add_argument('--lr', type=float, default=0.00001,
                help="learning rate for updating the distortion via stochastic gradient descent or Adam. Default: 0.001.")
            weather_args.add_argument('--optimizer', default="Adam",
                help="the optimizer used for the perturbations.")


            weather_args.add_argument('--learn_offset', default=True, type=ast.literal_eval,
                help="if specified, initial position of the particles will be optimized.")
            weather_args.add_argument('--learn_transparency', default=True, type=ast.literal_eval,
                help="if specified, the transparency of the particles will be optimized.")
            weather_args.add_argument('--learn_motionoffset', default=True, type=ast.literal_eval,
                help="if specified, the endpoint of the particle motion will be optimized (along with the starting point).")
            weather_args.add_argument('--learn_color', default=True, type=ast.literal_eval,
                help="if specified, the color of the particle will be optimized.")

            weather_args.add_argument('--alph_motion', default=1000., type=float,
                help="weighting for the motion loss.")
            weather_args.add_argument('--alph_motionoffset', default=1000., type=float,
                help="weighting for the motion offset loss.")

        if stage == "evaluation":
            pass

    return parser

def print_args_pretty(args, parser):
    arg_groups={}

    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=group_dict

    arg_groups.pop("positional arguments")
    arg_groups.pop("optional arguments")
    print(json.dumps(arg_groups, indent=2))
