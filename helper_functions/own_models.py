import torch
import torch.nn as nn
import logging

from helper_functions import ownutilities
from weather_attack.render import render


class ScaledInputWeatherModel(nn.Module):
    def __init__(self,  net, make_unit_input=False, **kwargs):
        super(ScaledInputWeatherModel, self).__init__()

        self.make_unit_input = make_unit_input
        self.model_name = net
        logging.info("Creating a Model with scaled input and the following parameters:")
        logging.info("\tmake_unit_input=%s" % (str(make_unit_input)))

        self.model_loaded, self.path_weights = ownutilities.import_and_load(net, **kwargs)


    def return_path_weights(self):
        return self.path_weights


    def forward(self, image1, image2, weather=None, scene_data=None, args_=None, test_mode=True, *args, **kwargs):
        if weather is not None:
            image1, image2 = render(image1, image2, scene_data, weather, args_)

        # If model expects images in [0,255], transform them from [0,1]
        if self.make_unit_input:
            image1 = 255.*image1
            image2 = 255.*image2

        # return self.model_loaded(image1, image2, *args, **kwargs)
        return ownutilities.compute_flow(self.model_loaded, self.model_name, image1, image2, test_mode=test_mode, *args, **kwargs)