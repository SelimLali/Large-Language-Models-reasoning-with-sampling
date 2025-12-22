#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
SEED=3
NUM_SHARDS=1
SHARD_ID=0

MAX_EXAMPLES=30
TOP_P=1.0

ALPHA=4
B=192
NMCMC_list=(1 2 3 4)

TOK_POWER=1024

RAW_ROOT="results/math500/raw_nmcmc_sweep"

echo "=== Sweep n_MCMC for POWER_MH | model=${MODEL} | seed=${SEED} | max_examples=${MAX_EXAMPLES} ==="
date

for n_mcmc in "${NMCMC_list[@]}"; do
  echo "[POWER_MH] n_MCMC=${n_mcmc}"

  OUTDIR="${RAW_ROOT}/nmcmc_${n_mcmc}"
  mkdir -p "${OUTDIR}"

  python scripts/generate_math500.py \
    --model "${MODEL}" \
    --method power_mh \
    --alpha "${ALPHA}" \
    --B "${B}" \
    --n_mcmc "${n_mcmc}" \
    --seed "${SEED}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_id "${SHARD_ID}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_new_tokens "${TOK_POWER}" \
    --top_p "${TOP_P}" \
    --use_chat_template \
    --results_dir "${OUTDIR}"
done


echo "=== ROBUST EVAL ==="
python scripts/evaluate_math500.py \
  --merged_csv "results/math500/raw_nmcmc_sweep/nmcmc_1/MATH500__Qwen_Qwen2.5-Math-1.5B-Instruct__power_mh__seed3__shard0of1.csv" \
  --out_dir "results/math500/raw_nmcmc_sweep/nmcmc_1" \
  --write_eval_csv

python scripts/evaluate_math500.py \
  --merged_csv "results/math500/raw_nmcmc_sweep/nmcmc_2/MATH500__Qwen_Qwen2.5-Math-1.5B-Instruct__power_mh__seed3__shard0of1.csv" \
  --out_dir "results/math500/raw_nmcmc_sweep/nmcmc_2" \
  --write_eval_csv

python scripts/evaluate_math500.py \
  --merged_csv "results/math500/raw_nmcmc_sweep/nmcmc_3/MATH500__Qwen_Qwen2.5-Math-1.5B-Instruct__power_mh__seed3__shard0of1.csv" \
  --out_dir "results/math500/raw_nmcmc_sweep/nmcmc_3" \
  --write_eval_csv

python scripts/evaluate_math500.py \
  --merged_csv "results/math500/raw_nmcmc_sweep/nmcmc_4/MATH500__Qwen_Qwen2.5-Math-1.5B-Instruct__power_mh__seed3__shard0of1.csv" \
  --out_dir "results/math500/raw_nmcmc_sweep/nmcmc_4" \
  --write_eval_csv
echo "=== END ==="