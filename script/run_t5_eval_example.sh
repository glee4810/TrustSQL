#!/usr/bin/env bash
set -euo pipefail

GPU="${CUDA_VISIBLE_DEVICES:-0}"

for dataset in atis advising ehrsql; do
  checkpoint="outputs/${dataset}_t5-large/checkpoint_best.pth.tar"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    echo "Set the load_model_path in T5/config-large/eval_${dataset}_test.yaml or place the checkpoint at the expected path." >&2
    exit 1
  fi
done

echo "ATIS"
python T5/main.py \
  --config T5/config-large/eval_atis_test.yaml \
  --output_file prediction_raw.json \
  --CUDA_VISIBLE_DEVICES "${GPU}"

echo "Advising"
python T5/main.py \
  --config T5/config-large/eval_advising_test.yaml \
  --output_file prediction_raw.json \
  --CUDA_VISIBLE_DEVICES "${GPU}"

echo "EHRSQL"
python T5/main.py \
  --config T5/config-large/eval_ehrsql_test.yaml \
  --output_file prediction_raw.json \
  --CUDA_VISIBLE_DEVICES "${GPU}"
