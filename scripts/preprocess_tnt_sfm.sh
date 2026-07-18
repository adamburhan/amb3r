#!/bin/bash
#SBATCH --job-name=preprocess_tnt_sfm
#SBATCH --output=/network/scratch/a/adam.burhan/logs/preprocess_tnt_sfm/%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-6

sequences=(
    Barn
    Caterpillar
    Church
    Courthouse
    Ignatius
    Meetingroom
    Truck
)

seq=${sequences[$SLURM_ARRAY_TASK_ID]}
data_root=$SCRATCH/datasets/tanks_and_temples
data_path=$data_root/$seq/images/
demo_name=$seq
results_path=$data_root/amb3r/$seq


repo_root=$HOME/repos/amb3r

module load cuda/11.8
source $repo_root/.venv/bin/activate
cd $repo_root
python sfm/run.py --data_path $data_path --demo_name $demo_name --results_path $results_path