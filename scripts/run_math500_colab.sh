#!/usr/bin/env bash
set -euo pipefail

#COMMAND TO RUN IN TERMINAL TO EXECUTE THIS SCRIPT BASH
# =========================
#!chmod +x run_math500_colab.sh
#!./run_math500_colab.sh | tee results/math500/colab_run.log
# =========================


# =========================
# Colab run config (MATH500)
# =========================
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"

SEEDS=(0 1 2)
NUM_SHARDS=1
SHARD_ID=0

MAX_EXAMPLES=100
TOP_P=1.0

ALPHA=4
B=192
NMCMC=3

TOK_BASE=1024
TOK_POWER=1024

echo "=== Colab run: MATH500 (subset=${MAX_EXAMPLES}) | model=${MODEL} | seeds=${SEEDS[*]} ==="
date

# ---------------------------
# 1) BASE
# ---------------------------
echo "=== BASE ==="
for seed in "${SEEDS[@]}"; do
  echo "[BASE] seed=${seed}"
  python scripts/generate_math500.py \
    --model "${MODEL}" \
    --method base \
    --seed "${seed}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_id "${SHARD_ID}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_new_tokens "${TOK_BASE}" \
    --top_p "${TOP_P}" \
    --use_chat_template
done

# ---------------------------
# 2) LOWTEMP (tau = 1/alpha)
# ---------------------------
echo "=== LOWTEMP ==="
for seed in "${SEEDS[@]}"; do
  echo "[LOWTEMP] seed=${seed}"
  python scripts/generate_math500.py \
    --model "${MODEL}" \
    --method lowtemp \
    --alpha "${ALPHA}" \
    --seed "${seed}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_id "${SHARD_ID}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_new_tokens "${TOK_BASE}" \
    --top_p "${TOP_P}" \
    --use_chat_template
done

# ---------------------------
# 3) POWER_MH (blockwise, paper-style)
# ---------------------------
echo "=== POWER_MH ==="
for seed in "${SEEDS[@]}"; do
  echo "[POWER_MH] seed=${seed}"
  python scripts/generate_math500.py \
    --model "${MODEL}" \
    --method power_mh \
    --alpha "${ALPHA}" \
    --B "${B}" \
    --n_mcmc "${NMCMC}" \
    --seed "${seed}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_id "${SHARD_ID}" \
    --max_examples "${MAX_EXAMPLES}" \
    --max_new_tokens "${TOK_POWER}" \
    --top_p "${TOP_P}" \
    --use_chat_template
done

# ---------------------------
# 4) MERGE + ROBUST EVAL + PASS@K
# ---------------------------
echo "=== MERGE ==="
python scripts/merge_math500.py --run_prefix MATH500 --output_csv results/math500/merged/merged.csv

echo "=== ROBUST EVAL ==="
python scripts/evaluate_math500.py \
  --merged_csv results/math500/merged/merged.csv \
  --out_dir results/math500/eval \
  --write_eval_csv

echo "=== PASS@K ==="
# 3 seeds => n up to 3 per (method, example_id) => max_k=3
python scripts/passk_math500.py \
  --folder results/math500/raw \
  --run_contains MATH500 \
  --max_k 3 \
  --out_dir results/math500/eval

echo "=== DONE ==="
date