#!/usr/bin/env bash
# HealthBench training with golden-rubric rewards (similarity + reflect on golden + discrimination + diversity).
# Training flow: adversarial.training_flow=unified — rollout answers/rubrics, compute rewards, update both LoRAs.
# reflect_use_golden_rubrics=true avoids self-evaluation reward hacking.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."
PROJECT_ROOT="$(pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
fi

for proxy_var in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy; do
    unset "${proxy_var}"
done

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/evorubric-main:${PROJECT_ROOT}/third_party/verl:${PROJECT_ROOT}/rubric_guidance:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_TMPDIR="${RAY_TMPDIR:-${PROJECT_ROOT}/.ray_tmp}"
mkdir -p "${RAY_TMPDIR}"

MINILM_CACHE="${PROJECT_ROOT}/minilm-cache"
if [ -d "${MINILM_CACHE}/hub" ]; then
    export HF_HUB_CACHE="${MINILM_CACHE}/hub"
    export HF_HUB_OFFLINE=1
    echo "Using local sentence-transformer cache: ${HF_HUB_CACHE}"
fi

MODEL_PATH="${MODEL_PATH:-/path/to/your/sft-model}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${PROJECT_ROOT}/data/healthbench_train_easy_4k.json}"
EVAL_TEST_FILE="${EVAL_TEST_FILE:-${PROJECT_ROOT}/data/healthbench_official_hard_1k.json}"

MODEL_NAME="$(basename "${MODEL_PATH}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-evorubric_unified_golden_${MODEL_NAME}_${TIMESTAMP}}"

RESUME_ENABLED="${RESUME_ENABLED:-false}"
if [ "${RESUME_ENABLED}" = "true" ]; then
    RESUME_STEP="${RESUME_STEP:-0}"
    RESUME_POLICY_LORA="${RESUME_POLICY_LORA:-}"
    RESUME_RUBRICS_LORA="${RESUME_RUBRICS_LORA:-}"
else
    RESUME_STEP=0
    RESUME_POLICY_LORA=""
    RESUME_RUBRICS_LORA=""
fi

LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
DEFAULT_LR="${DEFAULT_LR:-2e-5}"
POLICY_LLM_LR="${POLICY_LLM_LR:-2e-5}"
RUBRICS_GENERATOR_LR="${RUBRICS_GENERATOR_LR:-5e-6}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
NUM_RUBRICS="${NUM_RUBRICS:-4}"
NUM_ANSWERS="${NUM_ANSWERS:-4}"
SAVE_LORA_FREQ="${SAVE_LORA_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3584}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LOGPROB_TOKEN_LEN="${MAX_LOGPROB_TOKEN_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-flash_attention_2}"

SWANLAB_ENABLED="${SWANLAB_ENABLED:-false}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

ENABLE_SIMILARITY="${ENABLE_SIMILARITY:-true}"
ENABLE_DISCRIMINATION="${ENABLE_DISCRIMINATION:-true}"
ENABLE_DIVERSITY="${ENABLE_DIVERSITY:-true}"
ENABLE_REFLECT="${ENABLE_REFLECT:-true}"

SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-0.25}"
DISCRIMINATION_WEIGHT="${DISCRIMINATION_WEIGHT:-0.25}"
REFLECT_WEIGHT="${REFLECT_WEIGHT:-0.25}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-0.25}"
SIMILARITY_PENALTY_WEIGHT="${SIMILARITY_PENALTY_WEIGHT:-0.0}"
SIMILARITY_PENALTY_THRESHOLD="${SIMILARITY_PENALTY_THRESHOLD:-0.0}"
ENABLE_RUBRICS_SIMILARITY_FILTER="${ENABLE_RUBRICS_SIMILARITY_FILTER:-false}"
RUBRICS_SIMILARITY_FILTER_THRESHOLD="${RUBRICS_SIMILARITY_FILTER_THRESHOLD:-0.9}"
REFLECT_USE_GOLDEN_RUBRICS="${REFLECT_USE_GOLDEN_RUBRICS:-true}"

NUM_ITERATIONS_PER_QUERY="${NUM_ITERATIONS_PER_QUERY:-1}"
MAX_RUBRICS_PARSE_RETRIES="${MAX_RUBRICS_PARSE_RETRIES:-1}"
MAX_CONCURRENT_API_CALLS="${MAX_CONCURRENT_API_CALLS:-64}"
ANSWER_AWARE_NUM_REFS="${ANSWER_AWARE_NUM_REFS:-0}"
USE_UNIVERSAL_RUBRICS_SYSTEM_PROMPT="${USE_UNIVERSAL_RUBRICS_SYSTEM_PROMPT:-true}"
ENABLE_EXEMPLAR_TRICK="${ENABLE_EXEMPLAR_TRICK:-false}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-0.7}"
REFLECT_GENERATION_TEMPERATURE="${REFLECT_GENERATION_TEMPERATURE:-0}"
DATASET_MODE="${DATASET_MODE:-open_rubrics}"
TRAINING_FLOW="${TRAINING_FLOW:-unified}"
POLICY_UPDATE_FREQ="${POLICY_UPDATE_FREQ:-1}"
RUBRICS_UPDATE_FREQ="${RUBRICS_UPDATE_FREQ:-1}"
POLICY_GRAD_ACCUM_STEPS="${POLICY_GRAD_ACCUM_STEPS:-1}"
RUBRICS_GRAD_ACCUM_STEPS="${RUBRICS_GRAD_ACCUM_STEPS:-1}"

LLM_EVALUATOR_TYPE="${LLM_EVALUATOR_TYPE:-deepseek}"
export LLM_EVALUATOR_TYPE
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-http://127.0.0.1:8000/v1}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-EMPTY}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v3.2}"

EVAL_ENABLED="${EVAL_ENABLED:-true}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-1000}"
EVAL_BEFORE_TRAIN="${EVAL_BEFORE_TRAIN:-true}"
EVAL_FREQ="${EVAL_FREQ:-100}"

LOG_DIR="${PROJECT_ROOT}/training_logs"
mkdir -p "${LOG_DIR}"
TERMINAL_LOG_FILE="${LOG_DIR}/run_unified_golden_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "EvoRubrics unified + open_rubrics + golden-rubric rewards"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Experiment: ${EXP_NAME}"
echo "Train: ${TRAIN_DATA_PATH}"
echo "Eval: ${EVAL_TEST_FILE}"
echo "Rewards: similarity=${SIMILARITY_WEIGHT}, reflect=${REFLECT_WEIGHT}, disc=${DISCRIMINATION_WEIGHT}, div=${DIVERSITY_WEIGHT}"
echo "reflect_use_golden_rubrics: ${REFLECT_USE_GOLDEN_RUBRICS}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (N_GPUS_PER_NODE=${N_GPUS_PER_NODE})"
echo "Log: ${TERMINAL_LOG_FILE}"
echo "=========================================="

python3 -m main_shared_base \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=false \
    data.train_files="${TRAIN_DATA_PATH}" \
    data.val_files=null \
    data.train_batch_size="${EFFECTIVE_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    data.shuffle=false \
    data.dataset_mode="${DATASET_MODE}" \
    data.num_rubrics="${NUM_RUBRICS}" \
    data.num_answers="${NUM_ANSWERS}" \
    policy_model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank="${LORA_R}" \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.override_config.attn_implementation="${ATTENTION_IMPLEMENTATION}" \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    +actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    +actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    +actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_LOGPROB_TOKEN_LEN}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_LOGPROB_TOKEN_LEN}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_LOGPROB_TOKEN_LEN}" \
    actor_rollout_ref.actor.optim.lr="${DEFAULT_LR}" \
    +actor_rollout_ref.actor.optim.policy_llm_lr="${POLICY_LLM_LR}" \
    +actor_rollout_ref.actor.optim.rubrics_generator_lr="${RUBRICS_GENERATOR_LR}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="${EFFECTIVE_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=null \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${N_GPUS_PER_NODE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.free_cache_engine=false \
    +actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    +actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    +actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature="${GENERATION_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_micro_batch_size=null \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
    algorithm.use_kl_in_reward=false \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name='healthbench_unified_openrubrics_grpo' \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-null}" \
    trainer.save_lora_freq="${SAVE_LORA_FREQ}" \
    trainer.resume_step="${RESUME_STEP}" \
    actor_rollout_ref.model.policy_lora_path="${RESUME_POLICY_LORA}" \
    actor_rollout_ref.model.rubrics_lora_path="${RESUME_RUBRICS_LORA}" \
    trainer.swanlab.enabled="${SWANLAB_ENABLED}" \
    trainer.swanlab.api_key="${SWANLAB_API_KEY}" \
    custom_reward_function.path="${PROJECT_ROOT}/evorubric-main/custom_reward_fn.py" \
    custom_reward_function.name="adversarial_reward_fn_batch" \
    adversarial.enable_similarity="${ENABLE_SIMILARITY}" \
    adversarial.enable_discrimination="${ENABLE_DISCRIMINATION}" \
    adversarial.enable_diversity="${ENABLE_DIVERSITY}" \
    adversarial.similarity_weight="${SIMILARITY_WEIGHT}" \
    adversarial.discrimination_weight="${DISCRIMINATION_WEIGHT}" \
    adversarial.diversity_weight="${DIVERSITY_WEIGHT}" \
    adversarial.enable_reflect="${ENABLE_REFLECT}" \
    adversarial.reflect_weight="${REFLECT_WEIGHT}" \
    adversarial.similarity_penalty_weight="${SIMILARITY_PENALTY_WEIGHT}" \
    adversarial.similarity_penalty_threshold="${SIMILARITY_PENALTY_THRESHOLD}" \
    adversarial.reflect_use_golden_rubrics="${REFLECT_USE_GOLDEN_RUBRICS}" \
    adversarial.num_iterations_per_query="${NUM_ITERATIONS_PER_QUERY}" \
    adversarial.max_rubrics_parse_retries="${MAX_RUBRICS_PARSE_RETRIES}" \
    adversarial.training_flow="${TRAINING_FLOW}" \
    adversarial.generation_temperature="${GENERATION_TEMPERATURE}" \
    adversarial.reflect_generation_temperature="${REFLECT_GENERATION_TEMPERATURE}" \
    adversarial.policy_update_freq="${POLICY_UPDATE_FREQ}" \
    adversarial.rubrics_update_freq="${RUBRICS_UPDATE_FREQ}" \
    adversarial.policy_grad_accum_steps="${POLICY_GRAD_ACCUM_STEPS}" \
    adversarial.rubrics_grad_accum_steps="${RUBRICS_GRAD_ACCUM_STEPS}" \
    adversarial.enable_exemplar_trick="${ENABLE_EXEMPLAR_TRICK}" \
    adversarial.answer_aware_num_refs="${ANSWER_AWARE_NUM_REFS}" \
    adversarial.use_universal_rubrics_system_prompt="${USE_UNIVERSAL_RUBRICS_SYSTEM_PROMPT}" \
    +adversarial.enable_rubrics_similarity_filter="${ENABLE_RUBRICS_SIMILARITY_FILTER}" \
    +adversarial.rubrics_similarity_filter_threshold="${RUBRICS_SIMILARITY_FILTER_THRESHOLD}" \
    +adversarial.max_concurrent_api_calls="${MAX_CONCURRENT_API_CALLS}" \
    trainer.eval.enabled="${EVAL_ENABLED}" \
    trainer.eval.test_file="${EVAL_TEST_FILE}" \
    trainer.eval.num_samples="${EVAL_NUM_SAMPLES}" \
    trainer.eval.eval_before_train="${EVAL_BEFORE_TRAIN}" \
    trainer.eval.eval_freq="${EVAL_FREQ}" \
    "$@" 2>&1 | tee "${TERMINAL_LOG_FILE}"

echo "log file: ${TERMINAL_LOG_FILE}"
