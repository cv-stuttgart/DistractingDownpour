#!/bin/bash
mkdir -p ../models/_pretrained_weights
# download weights for flownet2 and store under pretrained
FILENAME="FlowNet2_checkpoint.pth.tar"
FILEID="1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt
mv  $FILENAME ../models/_pretrained_weights
