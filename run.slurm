#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH -o ./slurm_out/slurm%A_%a_out.log
#SBATCH -e ./slurm_out/slurm%A_%a_err.log
#SBATCH --array=0-9%5

cmdlines=(
    "python <your_path_here>/codes_py/main.py --d "<your_path_here>/datasets/mnist" --n 100 --batch-size 64 --ms 16 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "<your_path_here>/codes_py/output/" "

    "python <your_path_here>/codes_py/main.py --d "<your_path_here>/datasets/mnist" --n 100 --batch-size 64 --ms 32 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "<your_path_here>/codes_py/output/" "

    "python <your_path_here>/codes_py/main.py --d "<your_path_here>/datasets/mnist" --n 100 --batch-size 64 --ms 64 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "<your_path_here>/codes_py/output/" "

    "python <your_path_here>/codes_py/main.py --d "<your_path_here>/datasets/mnist" --n 100 --batch-size 64 --ms 128 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "<your_path_here>/codes_py/output/" "

    "python <your_path_here>/codes_py/main.py --d "<your_path_here>/datasets/mnist" --n 100 --batch-size 64 --ms 256 --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "<your_path_here>/codes_py/output/" "

    )
cmd=${cmdlines[SLURM_ARRAY_TASK_ID]}
$cmd