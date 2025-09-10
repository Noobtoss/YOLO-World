#!/bin/bash
#SBATCH --job-name=YOLOX_train        # Kurzname des Jobs
#SBATCH --output=logs/R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                # RAM pro CPU Kern #20G #32G #64G

BASE_DIR=/nfs/scratch/staff/schmittth/codeNexus/YOLO-World
CFG=${1:-custom/configs/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py}
CKPT=${2:-models/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth}
IMGS=${3:-demo/sample_images}
PROMPTS=${4:-'person,dog,cat'}

export PYTHONPATH=$BASE_DIR:$PYTHONPATH
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda-YOLO-World

srun python custom/python_scripts/image_demo.py $BASE_DIR/$CFG $BASE_DIR/$CKPT $BASE_DIR/$IMGS $PROMPTS --topk 100 --threshold 0.005 --output-dir runs
