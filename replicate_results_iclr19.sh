# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
#! /bin/bash

# Usage ./replicate_results.sh <DATASET-NAME> <THREAD-ID> <JE>
## where <DATASET-NAME>: MNIST, CIFAR, CUB, AWA 

DATASET=$1
THREAD=$2
JE=$3
OPTIM='SGD'
BATCH_SIZE=10
if [ $DATASET = "MNIST" ]; then
    NUM_RUNS=5
    if [ $THREAD = 1 ]; then
        IMP_METHODS=( 'VAN' 'EWC' 'PI' )
        LRS=(0.03 0.03 0.1)
        LAMDAS=(0 10 0.1)
    elif [ $THREAD = 2 ]; then
        IMP_METHODS=( 'MAS' 'RWALK' )
        LRS=(0.1 0.1)
        LAMDAS=(0.1 1)
    elif [ $THREAD = 3 ]; then
        IMP_METHODS=( 'PNN' 'A-GEM' )
        LRS=(0.1 0.1)
        LAMDAS=(0 0)
    fi
    ARCH='FC-S'
    RESULTS_DIR='results/mnist'
    if [ ! -d $RESULTS_DIR ]; then
        mkdir -pv $RESULTS_DIR
    fi

    for ((i=0;i<${#IMP_METHODS[@]};++i)); do
        imp_method="${IMP_METHODS[i]}"
        lr=${LRS[i]}
        lam=${LAMDAS[i]}
        python ./fc_permute_mnist.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method $imp_method --synap-stgth $lam --log-dir $RESULTS_DIR
    done
elif [ $DATASET = "CIFAR" ]; then
    NUM_RUNS=5
    if [ $THREAD = 1 ]; then
        IMP_METHODS=( 'VAN' 'EWC' 'PI' )
        LRS=(0.01 0.03 0.01)
        LAMDAS=(0 10 0.1)
    elif [ $THREAD = 2 ]; then
        IMP_METHODS=( 'MAS' 'RWALK' )
        LRS=(0.03 0.03)
        LAMDAS=(0.1 1)
    elif [ $THREAD = 3 ]; then
        IMP_METHODS=( 'PNN' 'A-GEM' )
        LRS=(0.03 0.03)
        LAMDAS=(0 0)
    fi
    ARCH='RESNET-S'
    RESULTS_DIR='results/cifar'
    if [ ! -d $RESULTS_DIR ]; then
        mkdir -pv $RESULTS_DIR
    fi

    for ((i=0;i<${#IMP_METHODS[@]};++i)); do
        imp_method="${IMP_METHODS[i]}"
        lr=${LRS[i]}
        lam=${LAMDAS[i]}
        python ./conv_split_cifar.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method $imp_method --synap-stgth $lam --log-dir $RESULTS_DIR
    done
elif [ $DATASET = "CUB" ]; then
    NUM_RUNS=10
    if [ $THREAD = 1 ]; then
        IMP_METHODS=( 'VAN' 'EWC' )
    elif [ $THREAD = 2 ]; then
        IMP_METHODS=( 'PI' 'MAS' )
    elif [ $THREAD = 3 ]; then
        IMP_METHODS=( 'RWALK' 'A-GEM' )
    fi
    ARCH='RESNET-B'
    OHOT_RESULTS_DIR='results/cub/ohot'
    JE_RESULTS_DIR='results/cub/je'
    if [ ! -d $OHOT_RESULTS_DIR ]; then
        mkdir -pv $OHOT_RESULTS_DIR
    fi
    if [ ! -d $JE_RESULTS_DIR ]; then
        mkdir -pv $JE_RESULTS_DIR
    fi

    for ((i=0;i<${#IMP_METHODS[@]};++i)); do
        imp_method="${IMP_METHODS[i]}"
        if [ $JE = 0 ]; then
            python conv_split_cub.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --imp-method $imp_method --data-dir ./CUB_data/CUB_200_2011/images/ --log-dir $OHOT_RESULTS_DIR
        elif [ $JE = 1 ]; then
            python conv_split_cub_hybrid.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --imp-method $imp_method --data-dir ./CUB_data/CUB_200_2011/images/ --log-dir $JE_RESULTS_DIR
        fi
    done
elif [ $DATASET = "AWA" ]; then
    NUM_RUNS=10
    if [ $THREAD = 1 ]; then
        IMP_METHODS=( 'VAN' )
    elif [ $THREAD = 2 ]; then
        IMP_METHODS=( 'EWC' )
    elif [ $THREAD = 3 ]; then
        IMP_METHODS=( 'PI' )
    elif [ $THREAD = 4 ]; then
        IMP_METHODS=( 'RWALK' )
    elif [ $THREAD = 5 ]; then
        IMP_METHODS=( 'A-GEM' )
    fi
    ARCH='RESNET-B'
    OHOT_RESULTS_DIR='results/awa/ohot'
    JE_RESULTS_DIR='results/awa/je'
    if [ ! -d $OHOT_RESULTS_DIR ]; then
        mkdir -pv $OHOT_RESULTS_DIR
    fi
    if [ ! -d $JE_RESULTS_DIR ]; then
        mkdir -pv $JE_RESULTS_DIR
    fi

    for ((i=0;i<${#IMP_METHODS[@]};++i)); do
        imp_method="${IMP_METHODS[i]}"
        if [ $JE = 0 ]; then
            python conv_split_awa.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --imp-method $imp_method --data-dir ./AWA_data/Animals_with_Attributes2/ --log-dir $OHOT_RESULTS_DIR
        elif [ $JE = 1 ]; then
            python conv_split_awa_hybrid.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --imp-method $imp_method --data-dir ./AWA_data/Animals_with_Attributes2/ --log-dir $JE_RESULTS_DIR
        fi
    done
fi
