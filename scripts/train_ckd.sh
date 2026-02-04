set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXP_PATH="${SCRIPT_DIR}/ckd"

ROOT_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p $EXP_PATH
mkdir -p $EXP_PATH/tmp_checkpoints
mkdir -p $EXP_PATH/tensorboard

SCRIPT_ABS_PATH="$(realpath "${BASH_SOURCE[0]}")"
cp $SCRIPT_ABS_PATH $EXP_PATH/backup.sh

MODEL_PATH="${ROOT_DIR}/models/Qwen3-0.6B"
TEACHER_MODEL_PATH="${ROOT_DIR}/models/Teacher-8B"

REWARD_PATH="${ROOT_DIR}/simrl.py"
DATA_PATH="${ROOT_DIR}/rl_data.jsonl"
REF_DATA_PATH="${ROOT_DIR}/kd_data.jsonl"

ray job submit --no-wait --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"excludes":["/checkpoints/", "/tensorboard/", "/data/", "/logs/", "/.git/", "/experiments/"], "env_vars": {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1", "VLLM_USE_V1": "1"}}' \
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
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend nccl \
   --enable_prefix_caching \
   --enforce_eager \
   --vllm_enable_sleep \
   --prompt_max_len 5120 \
   --pretrain $MODEL_PATH \
   --teacher $TEACHER_MODEL_PATH \
   --remote_rm_url $REWARD_PATH \
   --save_path $EXP_PATH/checkpoints \
   --save_hf_ckpt \
   --load_checkpoint \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 1 \
   --prompt_max_len 5120 \
   --max_samples 100000000 \
   --generate_max_len 512 \
   --advantage_estimator dr_grpo \
   --zero_stage 1 \
   --gradient_checkpointing \
   --bf16 \
   --actor_learning_rate 3e-6 \
   --init_kl_coef 0.0 \
   --kd_kl_coef 1.0 \
   --op_kl_coef 0.0 \
   --ppo_coef 0.0 \
   --gamma 1.0 \
   --kd_ratio 1.0 \
   --kd_kl_l1_coef 10.0 \
   --l2 1e-4 \
   --use_kl_loss \
   --kd_kl_type fkl \
   --kl_estimator k2 \
   --ref_mask_type all \
   --prompt_data $DATA_PATH \
   --ref_prompt_data $REF_DATA_PATH \
   --input_key inputs \
   --normalize_reward \
   --packing_samples \
   --flash_attn \
   --save_steps 100 \
   --eval_steps -1 \
   --ckpt_path $EXP_PATH/tmp_checkpoints \
   --label_key targets \
   --ref_key targets \
   --token_level_kl \
   --use_tensorboard $EXP_PATH/tensorboard
