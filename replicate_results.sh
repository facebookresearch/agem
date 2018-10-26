#! /bin/bash

# Usage replicate_results.sh <DATASET-NAME>
## where <DATASET-NAME>: MNIST, CIFAR, CUB, AWA 

DATASET=$1
NUM_RUNS=5
OPTIM='SGD'
BATCH_SIZE=10
if [ $DATASET = "MNIST" ]; then
    IMP_METHODS=( 'VAN' 'EWC' 'PI' 'MAS' 'RWALK' 'PNN' 'A-GEM' )
    LRS=(0.03 0.03 0.01 0.1 0.1 0.1 0.1)
    LAMDAS=(0 10 0.1 0.1 1 0 0)
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
    IMP_METHODS=( 'VAN' 'EWC' 'PI' 'MAS' 'RWALK' 'PNN' 'A-GEM' )
    LRS=(0.01 0.03 0.01 0.03 0.03 0.03 0.03)
    LAMDAS=(0 10 0.1 0.1 1 0 0)
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
    IMP_METHODS=( 'VAN' 'EWC' 'PI' 'MAS' 'RWALK' 'A-GEM' )
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
        python conv_split_cub.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --imp-method $imp_method --data-dir ./CUB_data/CUB_200_2011/images/ --log-dir $OHOT_RESULTS_DIR
        python conv_split_cub_hybrid.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --imp-method $imp_method --data-dir ./CUB_data/CUB_200_2011/images/ --log-dir $JE_RESULTS_DIR
    done
elif [ $DATASET = "AWA" ]; then
    IMP_METHODS=( 'VAN' 'EWC' 'PI' 'MAS' 'RWALK' 'A-GEM' )
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
        python conv_split_awa_100.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --imp-method $imp_method --data-dir ./AWA_data/Animals_with_Attributes2/ --log-dir $OHOT_RESULTS_DIR
        python conv_split_awa_hybrid_100.py --train-single-epoch --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --imp-method $imp_method --data-dir ./AWA_data/Animals_with_Attributes2/ --log-dir $JE_RESULTS_DIR
    done
fi
