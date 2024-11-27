#!/bin/bash

magpie_root=/lustre/fswork/projects/rech/gkb/uvl55hq/magpie
cd ${magpie_root}/scripts_b

model_name_or_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/mistralai/Mistral-Large-Instruct-2411
sbatch gen_ins_c.slurm $model_name_or_path 500000

model_name_or_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/meta-llama/Meta-Llama-3.1-70B-Instruct
sbatch gen_ins_c.slurm $model_name_or_path 1000000
sbatch gen_ins_c.slurm $model_name_or_path 1000000

model_name_or_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
sbatch gen_ins_c.slurm $model_name_or_path 1000000
sbatch gen_ins_c.slurm $model_name_or_path 1000000

model_name_or_path=/lustre/fswork/projects/rech/gkb/commun/models/pretrained/Qwen/Qwen2.5-72B-Instruct
sbatch gen_ins_c.slurm $model_name_or_path 1000000
sbatch gen_ins_c.slurm $model_name_or_path 1000000


# 999971
# 999977