#! /bin/bash

# Usage ./download_cub_awa.sh

### Download the CUB and AWA datasets and setup the directories

CUB_DIR='CUB_data'
CUB_DNLD_FILE='CUB_200_2011.tgz'
CUB_PRETRAIN_DIR='resnet-18-pretrained-imagenet'
AWA_DIR='AWA_data'
AWA_DNLD_FILE='AwA2-data.zip'

# Download CUB dataset
if [ ! -d $CUB_DIR ]; then
    mkdir $CUB_DIR
fi
cd $CUB_DIR
if [ ! -f $CUB_DNLD_FILE ]; then
    wget -O $CUB_DNLD_FILE http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
fi
tar xzf $CUB_DNLD_FILE
cd ../

# Download ImageNet pretrained model for CUB
if [ ! -d $CUB_PRETRAIN_DIR ]; then
    mkdir $CUB_PRETRAIN_DIR
fi
cd $CUB_PRETRAIN_DIR
wget -O model.ckpt.data-00000-of-00001 https://www.dropbox.com/s/oea0jufizm55imo/model.ckpt.data-00000-of-00001?dl=0
wget -O model.ckpt.index https://www.dropbox.com/s/khsh0sjvqvrz6t1/model.ckpt.index?dl=0
wget -O model.ckpt.meta https://www.dropbox.com/s/dtb5qjv6i27tlyl/model.ckpt.meta?dl=0
cd ../

# Download AWA dataset
if [ ! -d $AWA_DIR ]; then
    mkdir $AWA_DIR
fi
cd $AWA_DIR
if [ ! -f $AWA_DNLD_FILE ]; then
    wget -O $AWA_DNLD_FILE https://cvml.ist.ac.at/AwA2/AwA2-data.zip
fi
unzip $AWA_DNLD_FILE
cd ../
