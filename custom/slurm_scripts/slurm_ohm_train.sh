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
CFG=${1:-custom/exps/Images04.py}
CKPT=${2:-models/yolox_x.pth}

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda_YOLO-World
