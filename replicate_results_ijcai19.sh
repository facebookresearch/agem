# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
#! /bin/bash

IMP_METHOD='ER'
NUM_RUNS=2
BATCH_SIZE=10
EPS_MEM_BATCH_SIZE=10
MEM_SIZE=1
LOG_DIR='results/'

python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --learning-rate 0.1 --imp-method $IMP_METHOD --synap-stgth 0 --log-dir $LOG_DIR --mem-size $MEM_SIZE --examples-per-task 1000 --eps-mem-batch $EPS_MEM_BATCH_SIZE

python conv_split_cifar.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --learning-rate 0.1 --imp-method $IMP_METHOD --synap-stgth 0 --log-dir $LOG_DIR --mem-size $MEM_SIZE --eps-mem-batch $EPS_MEM_BATCH_SIZE


