#/bin/bash
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
CUDA_VISIBLE_DEVICES=0,1

set -x
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    run.py \
    --exp-config simple_odometry_pointnav.yaml \
    --run-type train