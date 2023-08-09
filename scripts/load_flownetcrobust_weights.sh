#!/bin/bash
# download weights for flownetcrobust and store under pretrained 
FILENAME="RobustFlowNetC.pth"
URL_BASE="https://lmb.informatik.uni-freiburg.de/resources/binaries/cvpr22_adv_flow"

download () {
	wget --no-check-certificate "$URL_BASE/$1.pth"
}

# Robust FlowNetC
download RobustFlowNetC
mv  $FILENAME ../models/_pretrained_weights