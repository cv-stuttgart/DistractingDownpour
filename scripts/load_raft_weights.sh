#!/bin/bash
# download load weights for raft and store under pretrained weights
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip -d ../models/_pretrained_weights
mv ../models/_pretrained_weights/models/* ../models/_pretrained_weights
# clean up
rm models.zip
rm -r ../models/_pretrained_weights/models