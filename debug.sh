#!/bin/bash
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
CUDA_VISIBLE_DEVICES=None python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    run.py \
    --exp-config ppo_replay_pointnav.yaml \
    --run-type train