#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# Run magpie generation

set -x -e

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# slurm buffers python output by default, unbuffer python -u
export PYTHONUNBUFFERED="1"

# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# hf
# export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# export VLLM_ATTENTION_BACKEND="FLASHINFER"

model_path=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
total_prompts=${2:-1000}
ins_topp=${3:-1}
ins_temp=${4:-0.9}
res_topp=${5:-0.9}
res_temp=${6:-0.6}
res_rep=1
# device="4"
# tensor_parallel=1
# device="4,5,6,7"
# device="0,1,2,3"
# tensor_parallel=4
# device="0,1,2,3,4,5,6,7"
tensor_parallel=8
gpu_memory_utilization=0.95
n=200
batch_size=200

# Get Current Time
timestamp=$(date +%s)
# timestamp=$(date +%y%m%d-%H%M%S)

# Generate Pretty Name
job_name="${model_path##*/}_topp${ins_topp}_temp${ins_temp}_${timestamp}"

output_folder="/projects/bhuang/corpus/text/llm/generated/magpie"

### Setup Logging
# log_dir="data"
# if [ ! -d "../${log_dir}" ]; then
#     mkdir -p "../${log_dir}"
# fi

# job_path="../${log_dir}/${job_name}"

# mkdir -p $job_path
# exec > >(tee -a "$job_path/${job_name}.log") 2>&1
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] Pretty name: $job_name"
echo "[magpie.sh] Total Prompts: $total_prompts"
echo "[magpie.sh] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp"
echo "[magpie.sh] Response Generation Config: temp=$res_temp, top_p=$res_topp, rep=$res_rep"
echo "[magpie.sh] System Config: device=$device, n=$n, batch_size=$batch_size, tensor_parallel=$tensor_parallel"
echo "[magpie.sh] Timestamp: $timestamp"
echo "[magpie.sh] Job Name: $job_name"

echo "[magpie.sh] Start Generating Instructions..."
    # --sanitize \
    # --job_name $job_name \
    # --device $device \
python ../exp/gen_ins_b.py \
    --model_path $model_path \
    --total_prompts $total_prompts \
    --system_prompt \
    --top_p $ins_topp \
    --temp $ins_temp \
    --tensor_parallel $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --disable_early_stopping \
    --logits_processor \
    --n $n \
    --output_folder $output_folder \
    --timestamp $timestamp \
    --max_tokens 1024 \

echo "[magpie.sh] Finish Generating Instructions!"

# echo "[magpie.sh] Start Generating Responses..."
# CUDA_VISIBLE_DEVICES=$device python ../exp/gen_res.py \
#     --device $device \
#     --model_path $model_path \
#     --batch_size $batch_size \
#     --top_p $res_topp \
#     --temp $res_temp \
#     --rep $res_rep \
#     --tensor_parallel $tensor_parallel \
#     --gpu_memory_utilization $gpu_memory_utilization \
#     --input_file $job_path/Magpie_${model_path##*/}_${total_prompts}_${timestamp}_ins.json \
#     --offline \

# echo "[magpie.sh] Finish Generating Responses!"

echo "END TIME: $(date)"
