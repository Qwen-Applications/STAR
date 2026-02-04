set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXP_PATH="${SCRIPT_DIR}/sim_rl"

ROOT_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p $EXP_PATH
mkdir -p $EXP_PATH/tmp_checkpoints
mkdir -p $EXP_PATH/tensorboard

SCRIPT_ABS_PATH="$(realpath "${BASH_SOURCE[0]}")"
cp $SCRIPT_ABS_PATH $EXP_PATH/backup.sh

MODEL_PATH="${SCRIPT_DIR}/ckd/checkpoints"

REWARD_PATH="${ROOT_DIR}/simrl.py"
DATA_PATH="${ROOT_DIR}/rl_data.jsonl"

ray job submit --no-wait --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"excludes":["/checkpoints/", "/tensorboard/", "/data/", "/logs/", "/.git/"], "env_vars": {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "VLLM_USE_V1": "1"}}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.7 \
   --init_kl_coef 1e-3 \
   --enable_prefix_caching \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator group_norm \
   --pretrain $MODEL_PATH \
   --remote_rm_url $REWARD_PATH \
   --save_path $EXP_PATH/checkpoints \
   --ckpt_path $EXP_PATH/tmp_checkpoints \
   --save_steps 100 \
   --eval_steps -1 \
   --save_hf_ckpt \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 3 \
   --prompt_max_len 5120 \
   --max_samples 100000000 \
   --generate_max_len 512 \
   --zero_stage 1 \
   --flash_attn \
   --bf16 \
   --actor_learning_rate 3e-7 \
   --prompt_data $DATA_PATH \
   --input_key inputs \
   --label_key targets \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --use_tensorboard $EXP_PATH/tensorboard \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0.01 0.99 \
   --dynamic_filtering_min_diff 0
