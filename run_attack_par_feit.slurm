#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=gpu-a100-short

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5|dlg6

#SBATCH --job-name="test"
#SBATCH --account=punim1623
#SBATCH --time=0-12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=120G

### export WORLD_SIZE=2   ### update world size: nodes x ntasks-per-node
### export MASTER_PORT=28400
### echo ">>> NODELIST="${SLURM_NODELIST}
### master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
### export MASTER_ADDR=$master_addr
### echo ">>> MASTER_ADDR="$MASTER_ADDR

module purge

eval "$(conda shell.bash hook)"
conda activate anogpt

python -u attack_encoder_new.py \
  --z_note "ftrojan with 5% trainset, encoder_scope: 15-22" \
  --trigger ftrojan \
  --frac_per_class 0.05 \
  --epochs 1 \
  --encoder_scope 15 16 17 18 19 20 21 22 \
    

##Log this job's resource usage stats###
my-job-stats -a -n -s