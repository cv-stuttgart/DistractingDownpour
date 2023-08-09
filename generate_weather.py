from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from helper_functions import parsing_file, ownutilities, logging
from helper_functions.config_specs import Conf
from weather_attack.weather import get_weather, recolor_weather
from weather_attack.render import render


def toggle_vis(img1, img2):
    im1 = plt.imshow(img1)
    im2 = plt.imshow(img2)
    im2.set_visible(False)

    def toggle_images(event):
        if event.key != 't':
            return
        im1.set_visible(not im1.get_visible())
        im2.set_visible(not im2.get_visible())
        plt.draw()

    plt.connect('key_press_event', toggle_images)
    plt.show()


def main(args):
    print("Preparing data from %s %s\n" % (args.dataset, args.dataset_stage))
    data_loader, has_gt, has_cam, has_weather = ownutilities.prepare_dataloader(args, shuffle=False, get_weather=True)

    if not has_cam:
        print("The datset '%s' at stage '%s' does not contain information about the camera data, which is necessary for a weather attack. Please use a dataset that provides camera data.\nAborting." % (args.dataset, args.dataset_stage))
        exit()

    if Conf.config('useCPU') or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print("Setting Device to %s\n" % device)


    print("Starting weather generation on %s %s\n" % (args.dataset, args.dataset_stage))
    print(f"Initializing weather particles using {args.cpu_count} CPUs")
    for batch, datachunck in enumerate(tqdm(data_loader)):

        if has_weather:
          (image1, image2, _, _, flow, _, scene_data, extra) = datachunck
        else:
          (image1, image2, flow, _, scene_data, extra) = datachunck

        (root,), (split,), (seq,), (base,), (frame,), (weatherdat,) = extra
        if args.deterministic_startpos:
            seed = ownutilities.get_robust_seed(seq, frame.detach().item()) + args.deterministic_startpos_seed
        else:
            seed = None

        weather = get_weather(has_weather, weatherdat, scene_data, args, seed=seed)

        # Save the weather data to a data structure that resembles the loaded data
        if args.save_weather:
            logging.save_weather_sintelnaming(weather, root, split, seq, base, frame.detach().item(), args)

        # Otherwise visualize the weather data
        if not args.save_weather or args.save_images:

            image1 /= 255.0
            image2 /= 255.0

            image1, image2 = image1.to(device), image2.to(device)
            scene_data = [i.to(device) for i in scene_data]
            weather = [i.to(device) for i in weather]

            img1, img2 = render(image1, image2, scene_data, weather, args)

            if args.save_images:
                logging.save_image_sintelnaming(img1, root, split, seq, base+"1", frame.detach().item(), args)
                logging.save_image_sintelnaming(img2, root, split, seq, base+"2", frame.detach().item(), args)
            else:
                img1 = img1[0].permute(1,2,0).numpy()
                img2 = img2[0].permute(1,2,0).numpy()
                toggle_vis(img1, img2)

    print("Done generating weather.")

if __name__ == "__main__":
    parser = parsing_file.create_parser(stage='generation', attack_type='weather')
    args = parser.parse_args()
    main(args)