# Distracting Downpour

This repository contains the source code for the [ICCV'23 paper](https://arxiv.org/abs/2305.06716) **Distracting Downpour: Adversarial Weather Attacks for Motion Estimation** by Jenny Schmalfuss, Lukas Mehl and Andrés Bruhn.
If you find this work useful, please cite us as

> @InProceedings{Schmalfuss2023Distracting,<br>
> title     = {Distracting Downpour: Adversarial Weather Attacks for Motion Estimation},<br>
> author    = {Schmalfuss, Jenny and Mehl, Lukas and Bruhn, Andrés},<br>
> year      = {2023},<br>
> booktitle = {Proc. International Conference on Computer Vision (ICCV)},<br>
> publisher = {IEEE/CVF}<br>
> }

Our **generated datasets** will be [available here]().


# Initial setup

## Cloning the repo

Clone this repository and all its submodules:

```
git clone --recurse-submodules thisrepository
```

## External datasets

For the weather augmentations and attacks, we use the [*Sintel*](http://sintel.is.tue.mpg.de/) dataset.
Make sure to also download the depth and camera motion data from [here](http://sintel.is.tue.mpg.de/depth), and include it into the `Sintel/training` folder.
The datasets are assumed to be in a similar layout as for training [RAFT](https://github.com/princeton-vl/RAFT#required-data):

```
├── datasets
    ├── Sintel
        ├── test
        ├── training
```
If you have it already saved somewhere else, you may link to the files with
```bash
mkdir datasets
cd datasets
ln -s /path/to/Sintel Sintel
```
or specify the path and names directly in `helper_functions/config_specs.py`.


## Reproduction datasets

We will provide all generated weather data and a small sample dataset as [download](). Note that the files are very large (particularly the `_npz` versions of the datasets). We therefore recommend to select what you need.

## Setup virtual environment

```shell
python3 -m venv weather_attack
source weather_attack/bin/activate
```

## Install required packages:
Change into scripts folder and execute the script which installs all required packages via pip. As each package is installed successively, you can debug errors for specific packages later.

```shell
cd scripts
bash install_packages.sh
```

## Loading flow models

Download the weights for a specific model by changing into the `scripts/` directory and executing the bash script for a specific model:
```shell
cd scripts
./load_[model]_weights.sh
```
Here `[model]` should be replaced by one of the following options:
```	
[ all | raft | gma | spynet | flownetcrobust | flownet2]
```
Note: the load_model scripts remove .git files, which are often write-protected and then require an additional confirmation on removal. To automate this process, consider to execute instead
```shell
yes | ./load_[model]_weights.sh
```
If you want to use FlowFormer, please download their checkpoints directly from their [website](https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_), and put the `sintel.pth` checkpoint into `models/_pretrained_weights/flowformer_weights/sintel.pth`

### Compiling Cuda extensions for FlowNet2

Please refer to the [pytorch documentation](https://github.com/NVIDIA/flownet2-pytorch.git) how to compile the channelnorm, correlation and resample2d extensions.
If all else fails, go to the extension folders `/models/FlowNet/{channelnorm,correlation,resample2d}_package`, manually execute
```
python3 setup.py install
```
and potentially replace `cxx_args = ['-std=c++11']` by `cxx_args = ['-std=c++14']`, and the list of `nvcc_args` by `nvcc_args = []` in every setup.py file.
If manually compiling worked, you may need to add the paths to the respective .egg files in the `{channelnorm,correlation,resample2d}.py files`, e.g. for channelnorm via
```python
sys.path.append("/lib/pythonX.X/site-packages/channelnorm_cuda-0.0.0-py3.6-linux-x86_64.egg")
import channelnorm_cuda
```
The `site-packages` folder location varies depending on your operation system and python version. 



# Code Usage

The standard process to create and attack weather consists of three steps:

1. weather particle *generation* with [`generate_weather.py`](generate_weather.py) (may save particle data and / or rendered images)
2. weather augmentation *evaluation* with [`evaluate_weather.py`](evaluate_weather.py) (uses rendered images)
3. weather *attack* with [`attack_weather.py`](attack_weather.py) (needs particle data)

Note that generating particles may take a significant amount of time, as each particle is sampled in a 3D box and rejected if it is invisible in both of the frames.
The length of this process depends on the scene scale and amount of sampled particles.
Therefore it is highly recommended to first create particles and rendered weather images and, as second step, to evaluate the networks on them or attack with them.


# Weather particle generation

To generate particles with our framework, execute

```
python3 generate_weather.py --weather_tag my_weather --save_images True --save_weather True
```

**Weather saving, loading and reproducibility.**
The used arguments steer the saving behavior, while arguments to steer the particle properties are described below:
```
--weather_path                  | weather data will be saved to that path. If weather is generated for Sintel/training, and no weather_path is specified, a new folder will be created in Sintel/training/
--weather_tag                   | weather is created in weather_path, in a folder that is called weather or weather_<weather_tag> if a weather_tag was specified
--save_images=True              | If True, the rendered .png images of the particles in the Sintel scenes will be saved
--save_weather=True             | If True, the particle data will be saved as .npz

--deterministic_startpos=False         | If true, all random variables will be seeded
--deterministic_startpos_seed=0        | The seed for random variable seeding
--weather_data                         | A folder where previously generated weather data can be found. Use e.g. for first creating particles as .npz and then render to .png in a second step.
--cpu_count                            | an integer specifying the number of CPUs for parallel particle sampling. Recommendation: 8
```

**Particle base properties.**
Then, more arguments can be added to specify the generated weather properties (not the rendering):

```
--num_flakes=1000                         | int, number of particles to generate in frame pair
--flakesize_max=71                        | int, uneven, the maximal size per particle
--depth_decay=9                           | an exponential decay parameter that reduces the size for background particles.
--flake_template_folder=[dust,particles]  | the folder from which particle billboards are sampled. Particles work well for snow, rain and sparks, while dust is for fog.
--constant_transparency=0                 | if == 0, the transparency is a hat-function that peaks at a depth of 2. Otherwise, it is the default transparency for all initialized particles.

--motion_x=0.0                            | float, the x motion for each particle
--motion_y=0.0                            | float, the y motion for each particle
--motion_z=0.0                            | float, the z motion for each particle
--motion_random_scale=0.0                 | float, randomizes the magnitude of the particle motion relative to the motion vector length
--motion_random_angle=0.0                 | float, maximal random offset angle for the particle motion in degree. (default=0.0, max=180)

--flake_r=255                             | int, R value for particle RGB. (min=0, max=255)
--flake_g=255                             | int, G value for particle RGB. (min=0, max=255)
--flake_b=255                             | int, B value for particle RGB. (min=0, max=255)
--flake_random_h=0.0                      | float, randomized offset to H-value of particle color in HSL space. (min=0, max=380)
--flake_random_s=0.0                      | float, randomized offset to S-value of particle color in HSL space. (min=0, max=1)
--flake_random_l=0.0                      | float, randomized offset to L-value of particle color in HSL space. (min=0, max=1)
```

**Particle rendering.**
Finally, these arguments should be specified if the weather should be rendered:

```
--rendering_method=[additive,Meshkin]     | the rendering method for the particle color
--transparency_scale=1.0                  | a global transparency scale
--depth_check=False                       | if True, particles behind objects will not be rendered
--do_motionblur=True                      | if True, particles will have motion blur. Attention: increases rendering time by the number of motionblur_samples
--motionblur_scale=0.025                  | length of shown motion blur relative to motion length. (min=0, max=1)
--motionblur_samples=10                   | number of particles per initial particle to create motion blur.
```

**Dataset cropping**.
Potentially, creating particles for the full dataset is not desired. These arguments help to augment reduced dataset version. Make sure to use the same dataset specifications for downstream evaluation / attacks on the reduced augmented data.

```
--dstype=final                            | Sintel dataset type, either final or clean
--single_scene                            | only use single Sintel scene. Pass name of scene folder, e.g. "mountain_1"
--from_scene                              | use frames from certain scene onward. Pass name of scene folder where to start, e.g. "mountain_1"
--frame_per_scene                         | number of frame pairs per Sintel scene, take all if set to 0
--from_frame                              | for all scenes, the dataset starts from specified frame number. 0 takes all frames, 1 all except first, etc.

```



## Generating datasets for snow, rain, sparks and fog

Attention: The functions below will save your weather data to the location of your Sintel data, and the produced output files are potentially large. Specify a `--weather_path` if you want to save the output files to a custom location.

Snow:
```
python3 generate_weather.py --weather_tag snow --save_images True --save_weather True --num_flakes 3000 --flakesize_max 71 --depth_decay 9 --motion_y 0.2 --flake_template_folder particles --flake_r 255 --flake_g 255 --flake_b 255 --rendering_method additive --transparency_scale 0.75 --do_motionblur False --depth_check True --cpu_count 8
```

Rain:
```
python3 generate_weather.py --weather_tag rain --save_images True --save_weather True --num_flakes 3000 --flakesize_max 51 --depth_decay 9 --motion_y 0.2 --motion_random_scale 0.1 --motion_random_angle 4 --flake_template_folder particles --flake_r 255 --flake_g 255 --flake_b 255 --rendering_method additive --transparency_scale 0.75 --do_motionblur True --motionblur_scale 0.15 --motionblur_samples 20 --depth_check True --cpu_count 8
```

Sparks:
```
python3 generate_weather.py --weather_tag sparks --save_images True --save_weather True --num_flakes 3000 --flakesize_max 41 --depth_decay 9 --motion_y -0.05 --motion_random_scale 0.2 --motion_random_angle 60 --flake_template_folder particles --flake_r 255 --flake_g 100 --flake_b 70 --flake_random_h 15 --flake_random_l 0.1 --rendering_method additive --transparency_scale 1.5 --do_motionblur True --motionblur_scale 0.3 --motionblur_samples 10 --depth_check True --cpu_count 8
```

Fog:
```
python3 generate_weather.py --weather_tag fog --save_images True --save_weather True --num_flakes 60 --flakesize_max 451 --depth_decay 0.8 --constant_transparency 0.3 --motion_y 0.0 --flake_template_folder dust --flake_r 255 --flake_g 255 --flake_b 255 --rendering_method additive --transparency_scale 0.25 --do_motionblur False --depth_check True --cpu_count 8
```


## Dataset samples - structure

We will provide samples of the rendered augmented images as [download]() for snow (`particles_3000_png`), rain (`rain_1500_png`), sparks (`color_additive_red_png`) and fog (`size_fog_png`) on the first frame pair per sequence from scene market_2 onward, which corresponds to executing the weather generation with `--frame_per_scene 1 --from_scene market_2`.

```
├── Weather_sampledata
    ├── particles_3000_png
        ├── market_2
            ├── frame_1_0001.png
            ├── frame_2_0001.png
        ├── market_5
            ├── frame_1_0001.png
            ├── frame_2_0001.png
        ├── ...
    ├── rain_1500_png
    ├── color_additive_red_png
    ├── size_fog_png
```




# Weather augmentation evaluation

Once the weather datasets have been created and the weather is rendered, the optical flow methods can be evaluated on the rendered augmented frames.

```
python3 evaluate_weather.py --weather_data path/to/weather/data --net network

```

Here, only the weather data and the network need to be specified:
```
--weather_data                                                | path to previously rendered weather, e.g. MPI-Sintel/training/weather_snow, if "--weather_tag snow" was used to generate weather. 
--net = [RAFT,GMA,FlowFormer,SpyNet,FlowNet2,FlowNetCRobust]  | choose a network to test with.

```

Also, use **dataset cropping** arguments from above as needed.
Logs over all metrics are generated with mlflow (see below), and a summary over the robustness is printed at the evaluation end.

You may test your method with all weather variations from the paper, by downloading the `_png` versions of our [dataset here]().


## Evaluation with the provided data-samples

Here, we show the exemplary calls for evaluating the provided weather augmentation samples (from `weather_sampledata`, which is provided as download) on GMA:

```
python3 evaluate_weather.py --weather_data Weather_sampledata/particles_3000_png --net GMA --frame_per_scene 1 --from_scene market_2
python3 evaluate_weather.py --weather_data Weather_sampledata/rain_1500_png --net GMA --frame_per_scene 1 --from_scene market_2
python3 evaluate_weather.py --weather_data Weather_sampledata/color_additive_red_png --net GMA --frame_per_scene 1 --from_scene market_2
python3 evaluate_weather.py --weather_data Weather_sampledata/size_fog_png --net GMA --frame_per_scene 1 --from_scene market_2
```



# Weather attack

Now, the generated particle files (`.npz`) can serve as basis to adversarially optimize particle properties:

```
python3 attack_weather.py --weather_data path/to/weather/data --net network
```

**Attack arguments**.
Use `--weather_data` and `--net` as explained above. Now, also attack arguments can be specified:
```
--steps=750                               | Adam optimization steps
--lr=0.00001                              | Adam learning rate

--alph_motion=1000                        | loss parameter alpha_1
--alph_motionoffset=1000                  | loss parameter alpha_2
--depth_check_differentiable=False        | if True, the depth check will be made differentiable

--learn_offset=True                       | if True, the initial position is optimized
--learn_motionoffset=True                 | if True, the offset to the motion vector is optimized
--learn_color=True                        | if True, the color is optimized
--learn_transparency=True                 | if True, the transparency is optimized

```

Also, use **dataset cropping** and **Particle rendering** arguments from above as needed, and keep the particle rendering arguments that were used to create the weather.
Logs over all metrics are generated with mlflow (see below), and a summary over the robustness is printed at the evaluation end.


## Attacking GMA with snow, rain, sparks and fog

Please generate datasets for snow, rain, sparks and fog as described above, or download the full .npz samples. If you [downloaded]() the `npz` samples, replace the `--weather_data` filenames as follows: `weather_snow` by `particles_3000_png`, `weather_rain` by `rain_1500_png`, `weather_sparks` by `color_additive_red_png` and `weather_fog` by `size_fog_png`

Snow:
```
python3 attack_weather.py --weather_data datafolder/weather_snow --net GMA --num_flakes 3000 --flakesize_max 71 --depth_decay 9 --motion_y 0.2 --flake_template_folder particles --flake_r 255 --flake_g 255 --flake_b 255 --rendering_method additive --transparency_scale 0.75 --do_motionblur False --depth_check True
```

Rain:
```
python3 attack_weather.py --weather_data datafolder/weather_rain --net GMA --num_flakes 3000 --flakesize_max 51 --depth_decay 9 --motion_y 0.2 --motion_random_scale 0.1 --motion_random_angle 4 --flake_template_folder particles --flake_r 255 --flake_g 255 --flake_b 255 --rendering_method additive --transparency_scale 0.75 --do_motionblur True --motionblur_scale 0.15 --motionblur_samples 20 --depth_check True
```

Sparks:
```
python3 attack_weather.py --weather_data datafolder/weather_sparks --net GMA --num_flakes 3000 --flakesize_max 41 --depth_decay 9 --motion_y -0.05 --motion_random_scale 0.2 --motion_random_angle 60 --flake_template_folder particles --flake_r 255 --flake_g 100 --flake_b 70 --flake_random_h 15 --flake_random_l 0.1 --rendering_method additive --transparency_scale 1.5 --do_motionblur True --motionblur_scale 0.3 --motionblur_samples 10 --depth_check True
```

Fog:
```
python3 attack_weather.py --weather_data datafolder/weather_fog --net GMA --learn_motionoffset False --num_flakes 60 --flakesize_max 451 --depth_decay 0.8 --constant_transparency 0.3 --motion_y 0.0 --flake_template_folder dust --flake_r 255 --flake_g 255 --flake_b 255 --rendering_method additive --transparency_scale 0.25 --do_motionblur False --depth_check True --depth_check_differentiable True
```




# Data logging and progress tracking

Training progress and output images are tracked with MLFlow in `mlruns/`, and output images and flows are additionally saved in `experiment_data/`.
In `experiment_data/`, the folder structure is `<networkname>_<attacktype>_<perturbationtype>/`, where each subfolder contains different runs of the same network with a specific perturbation type.

To view the mlflow data locally, navigate to the root folder of this repository, execute

```shell
mlflow ui

```
and follow the link that is displayed. This leads to the web interface of mlflow.

If the data is on a remote host, the below procedure will get the mlflow data displayed.

## Progress tracking with MLFlow (remote server)

Identify the remote's public IP address via

```shell
curl ifconfig.me
```
then start mlflow on remote machine:
```shell
mlflow server --host 0.0.0.0
```
On your local PC, replace 0.0.0.0 with the public IP and visit the following address in a web-browser:
```shell
http://0.0.0.0:5000
```



# External models and dependencies

## Models 
- [*FlowFormer*](https://github.com/drinkingcoder/FlowFormer-Official)
- [*GMA*](https://github.com/zacjiang/GMA.git)
- [*RAFT*](https://github.com/princeton-vl/RAFT)
- [*SpyNet*](https://github.com/sniklaus/pytorch-spynet.git) and [Flow Attack](https://github.com/anuragranj/flowattack.git)
- [*FlowNet*](https://github.com/NVIDIA/flownet2-pytorch.git)
- [*FlowNetCRobust*](https://github.com/lmb-freiburg/understanding_flow_robustness)

## Additional code

- Attack structure and model handling from [PCFA](https://github.com/cv-stuttgart/PCFA)

- Augmentation and dataset handling (`datasets.py` `frame_utils.py` `InputPadder`) from [RAFT](https://github.com/princeton-vl/RAFT)

- Path configuration (`conifg_specs.py`) inspired by [this post](https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py)

- File parsing (`parsing_file.py`): idea from [this post](https://stackoverflow.com/a/60418265/13810868)
