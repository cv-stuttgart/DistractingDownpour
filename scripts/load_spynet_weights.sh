#!/bin/bash
# download load weights for gma and store under pretrained weights
git clone https://github.com/anuragranj/flowattack.git temp
mkdir -p ../models/_pretrained_weights/spynet_weights
mv temp/models/spynet_models/* ../models/_pretrained_weights/spynet_weights/
# clean up
rm -r temp