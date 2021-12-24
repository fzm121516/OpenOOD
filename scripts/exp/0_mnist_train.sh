#!/bin/bash
# sh scripts/exp/0_mnist_train.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} --kill-on-bad-exit=1 \
--job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/run.py \
--config configs/datasets/mnist_datasets.yml \
configs/networks/lenet.yml \
configs/train/mnist_baseline.yml \
--dataset.image_size 28 \
--network.name lenet
