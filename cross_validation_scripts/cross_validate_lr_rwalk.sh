#! /bin/bash
# Usage ./cross_validate_lr.sh <EXPERIMENT_NAME>, where <EXPERIMENT_NAME> could be 'SPLIT_MNIST', 'PERMUTE_MNIST', 'SPLIT_CIFAR'
set -e

EXP=$1
IMP_METHOD='RWALK'
SYNAP_STGTH=(0.1 1 10 100)
BATCH_SIZE=10
LOG_DIR='../cross_validation_results'
ARCH='RESNET'
OPTIM='SGD'
if [ $EXP = "SPLIT_MNIST" ]; then
    LR=(0.0001 0.001 0.003 0.01 0.03 0.1)
    for lamda in ${SYNAP_STGTH[@]}
    do
        for lr in ${LR[@]}
        do
            python ../fc_split_mnist.py --cross-validate-mode --train-single-epoch --num-runs 5 --batch-size $BATCH_SIZE --learning-rate $lr --imp-method $IMP_METHOD --synap-stgth $lamda --log-dir $LOG_DIR
        done
    done
elif [ $EXP = "PERMUTE_MNIST" ]; then
    LR=(0.0001 0.001 0.003 0.01 0.03 0.1)
    for lamda in ${SYNAP_STGTH[@]}
    do
        for lr in ${LR[@]}
        do
            python ../fc_permute_mnist.py --cross-validate-mode --train-single-epoch --num-runs 5 --batch-size $BATCH_SIZE --learning-rate $lr --imp-method $IMP_METHOD --synap-stgth $lamda --log-dir $LOG_DIR
        done
    done
elif [ $EXP = "SPLIT_CIFAR" ]; then
    LR=(0.001 0.003 0.01 0.03 0.1 0.3 1.0) 
    for lamda in ${SYNAP_STGTH[@]}
    do
        for lr in ${LR[@]}
        do
            python ../conv_split_cifar.py --cross-validate-mode --train-single-epoch --arch $ARCH --num-runs 2 --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method $IMP_METHOD --synap-stgth $lamda --log-dir $LOG_DIR
        done
    done
elif [ $EXP = "SPLIT_CUB_ONE_HOT" ]; then
    LR=(0.001) 
    for lamda in ${SYNAP_STGTH[@]}
    do
        for lr in ${LR[@]}
        do
            python ../conv_split_cub.py --train-single-epoch --cross-validate-mode --arch RESNET --num-runs 2 --optim SGD --learning-rate $lr --batch-size $BATCH_SIZE --imp-method $IMP_METHOD --synap-stgth $lamda --data-dir ../CUB_data/CUB_200_2011/images --init-checkpoint ../resnet-18-pretrained-imagenet/model.ckpt --log-dir $LOG_DIR
        done
    done
else
    echo "ERROR! Wrong Experiment Name!!"
    exit 1
fi

