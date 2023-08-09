#!/bin/bash
# download load weights for gma and store under pretrained weights
git clone https://github.com/zacjiang/GMA.git
mv GMA/checkpoints/* ../models/_pretrained_weights
# clean up
rm -r GMA