#/bin/bash
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

rm -rf odom_tb

set -x
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    run.py \
    --exp-config simple_odometry_pointnav.yaml \
    --run-type train

#export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
#
#set -x
#srun python -u -m run \
#    --exp-config ddppo_tamer_pointnav.yaml \
#    --run-type train