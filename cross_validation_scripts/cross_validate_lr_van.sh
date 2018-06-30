#! /bin/bash
# Usage ./cross_validate_lr.sh <EXPERIMENT_NAME>, where <EXPERIMENT_NAME> could be 'SPLIT_MNIST', 'PERMUTE_MNIST', 'SPLIT_CIFAR'
set -e

EXP=$1
IMP_METHOD='VAN'
SYNAP_STGTH=(0)
BATCH_SIZE=10
LOG_DIR='../cross_validation_results'
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
    LR=(0.00001 0.0001 0.001 0.003 0.01 0.03 0.1)
    for lamda in ${SYNAP_STGTH[@]}
    do
        for lr in ${LR[@]}
        do
            python ../conv_split_cifar.py --cross-validate-mode --train-single-epoch --num-runs 2 --batch-size $BATCH_SIZE --learning-rate $lr --imp-method $IMP_METHOD --synap-stgth $lamda --log-dir $LOG_DIR
        done
    done
else
    echo "ERROR! Wrong Experiment Name!!"
    exit 1
fi

