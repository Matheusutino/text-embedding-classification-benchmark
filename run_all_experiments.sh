#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
OUTPUT_DIR="${ROOT_DIR}/artifacts"
FORCE_RECOMPUTE=""

DATASETS=(
  "${ROOT_DIR}/datasets/CSTR.csv"
  "${ROOT_DIR}/datasets/Dmoz-Computers.csv"
  "${ROOT_DIR}/datasets/Dmoz-Health.csv"
  "${ROOT_DIR}/datasets/Dmoz-Science.csv"
  "${ROOT_DIR}/datasets/Dmoz-Sports.csv"
  "${ROOT_DIR}/datasets/NSF.csv"
  "${ROOT_DIR}/datasets/SyskillWebert.csv"
  "${ROOT_DIR}/datasets/classic4.csv"
  "${ROOT_DIR}/datasets/re8.csv"
  "${ROOT_DIR}/datasets/review_polarity.csv"
  "${ROOT_DIR}/datasets/sms_spam.csv"
)

REPRESENTATIONS=(
  "bow_unigram"
  "bow_unigram_bigram"
  "tfidf_unigram"
  "tfidf_unigram_bigram"
  "tfidf_char_wb_3_5"
  "all_minilm_l6_v2"
  "all_mpnet_base_v2"
  "multi_qa_minilm_l6_cos_v1"
  "e5_large_v2"
  "bge_m3"
  "gte_modernbert_base"
  "nomic_embed_text_v1_5"
  "embeddinggemma_300m"
  "granite_embedding_125m_english"
  "qwen3_embedding_0_6b"
  "qwen3_embedding_8b"
  "gemini_embedding_2_preview"
  "text_embedding_3_large"
)

if [[ "${1:-}" == "--force-recompute" ]]; then
  FORCE_RECOMPUTE="--force-recompute"
fi

slugify_dataset() {
  local dataset_name="$1"
  dataset_name="${dataset_name%.csv}"
  dataset_name="${dataset_name,,}"
  dataset_name="${dataset_name// /_}"
  dataset_name="${dataset_name//-/_}"
  echo "${dataset_name}"
}

should_skip_experiment() {
  local dataset_path="$1"
  local representation="$2"
  local dataset_name
  local dataset_slug
  local summary_path
  local status

  if [[ -n "${FORCE_RECOMPUTE}" ]]; then
    return 1
  fi

  dataset_name="$(basename "${dataset_path}")"
  dataset_slug="$(slugify_dataset "${dataset_name}")"
  summary_path="${OUTPUT_DIR}/runs/${representation}/${dataset_slug}/summary.json"

  if [[ ! -f "${summary_path}" ]]; then
    return 1
  fi

  status="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
path = Path(${summary_path@Q})
data = json.loads(path.read_text(encoding="utf-8"))
print(data.get("status", ""))
PY
)"

  [[ "${status}" == "ok" ]]
}

for dataset_path in "${DATASETS[@]}"; do
  for representation in "${REPRESENTATIONS[@]}"; do
    echo
    echo "dataset=$(basename "${dataset_path}") representation=${representation}"
    if should_skip_experiment "${dataset_path}" "${representation}"; then
      echo "skipped: summary.json already exists with status=ok"
      continue
    fi
    PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -m embeddings_pipeline \
      --dataset "${dataset_path}" \
      --representation "${representation}" \
      --output-dir "${OUTPUT_DIR}" \
      ${FORCE_RECOMPUTE}
  done
done
