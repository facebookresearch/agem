#! /bin/bash

# Usage ./download_cub_awa.sh

### Download the CUB and AWA datasets and setup the directories

CUB_DIR='CUB_data'
CUB_DNLD_FILE='CUB_200_2011.tgz'
AWA_DIR='AWA_data'
AWA_DNLD_FILE='AwA2-data.zip'

if [ ! -d $CUB_DIR ]; then
    mkdir $CUB_DIR
fi
cd $CUB_DIR
if [ ! -f $CUB_DNLD_FILE ]; then
    wget -O $CUB_DNLD_FILE http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
fi
tar xzf $CUB_DNLD_FILE
cd ../

if [ ! -d $AWA_DIR ]; then
    mkdir $AWA_DIR
fi
cd $AWA_DIR
if [ ! -f $AWA_DNLD_FILE ]; then
    wget -O $AWA_DNLD_FILE https://cvml.ist.ac.at/AwA2/AwA2-data.zip
fi
unzip $AWA_DNLD_FILE
cd ../
