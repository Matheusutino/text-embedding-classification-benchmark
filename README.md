# Embeddings Classification Pipeline

Simple text classification pipeline with reusable feature caches.

The project runs stratified cross-validation experiments over CSV datasets with `text` and `class` columns, supports both lexical features and dense embeddings, and stores cached representations on disk so they can be reused across runs.

## What It Does

- Loads one dataset at a time from `datasets/*.csv`
- Runs outer stratified 5-fold cross-validation
- Splits each outer training fold into stratified `train` and `val`
- Selects the best `LogisticRegression` hyperparameter with validation `f1_macro`
- Retrains on `train + val`
- Evaluates once on the held-out outer `test`
- Saves per-fold metrics and aggregated results
- Caches generated features in `artifacts/features/`

## Experiment Protocol

The current setup is fixed in code:

- Outer CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Inner split: stratified `train/val` with `val_size=0.2`
- Classifier: `LogisticRegression`
- Hyperparameter grid: `C = [0.01, 0.1, 1, 10]`
- Model selection metric: `f1_macro`
- Seed: `42`

For lexical features, the representation is fit only on the training data of the relevant stage:

- model selection: `fit_transform(train)` and `transform(val)`
- final evaluation: `fit_transform(train + val)` and `transform(test)`

This keeps the test fold isolated from feature construction and hyperparameter selection.

## Supported Representations

### Lexical

- `bow_unigram`
- `bow_unigram_bigram`
- `tfidf_unigram`
- `tfidf_unigram_bigram`
- `tfidf_char_wb_3_5`

All lexical variants use `max_features=1000`.

### Dense local models

- `all_minilm_l6_v2`
- `all_mpnet_base_v2`
- `multi_qa_minilm_l6_cos_v1`
- `e5_large_v2`
- `bge_m3`
- `gte_modernbert_base`
- `nomic_embed_text_v1_5`
- `embeddinggemma_300m`
- `granite_embedding_125m_english`
- `qwen3_embedding_0_6b`
- `qwen3_embedding_8b`

### Remote models via OpenRouter

- `gemini_embedding_2_preview`
- `text_embedding_3_large`

## Prefix Handling

Some models require task prefixes:

- `intfloat/e5-large-v2` uses `query: `
- `nomic-ai/nomic-embed-text-v1.5` uses `classification: `

## Project Structure

```text
src/embeddings_pipeline/
  __main__.py
  cli.py
  config.py
  dataset.py
  cache.py
  experiment.py
  representations.py

datasets/
artifacts/
tests/
run_all_experiments.sh
```

## Setup

Create a virtual environment and install the package:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

This installs the full runtime stack, including lexical features, dense embedding models, and OpenRouter support. The `dev` extra only adds test dependencies.

## Environment Variables

The CLI loads `.env` automatically.

For OpenRouter-backed models, set:

```env
OPENROUTER_API_KEY=your_key_here
```

Files included:

- `.env`
- `.env.example`

## Running One Experiment

Example with a lexical representation:

```bash
PYTHONPATH=src .venv/bin/python -m embeddings_pipeline \
  --dataset datasets/sms_spam.csv \
  --representation tfidf_unigram \
  --output-dir artifacts
```

Force recomputation:

```bash
PYTHONPATH=src .venv/bin/python -m embeddings_pipeline \
  --dataset datasets/sms_spam.csv \
  --representation tfidf_unigram \
  --output-dir artifacts \
  --force-recompute
```

## Running All Experiments

The repository includes a simple shell script with fixed dataset paths and representation names:

```bash
./run_all_experiments.sh
```

With cache bypass:

```bash
./run_all_experiments.sh --force-recompute
```

If you want to customize the sweep, edit the `DATASETS` and `REPRESENTATIONS` arrays directly in [run_all_experiments.sh](/home/matheus/Desktop/Itens./Itens/Projetos/embeddings-new/run_all_experiments.sh:1).

## Cache Layout

All cached representations live under `artifacts/features/`.

Examples:

```text
artifacts/features/tfidf_unigram/sms_spam/fold_1/train/
artifacts/features/tfidf_unigram/sms_spam/fold_1/val/
artifacts/features/tfidf_unigram/sms_spam/fold_1/train_val/
artifacts/features/tfidf_unigram/sms_spam/fold_1/test_final/

artifacts/features/sentence-transformers__all-minilm-l6-v2/sms_spam/full_dataset/
```

Each cache directory stores:

- feature file
  - lexical: `features.npz`
  - dense: `embeddings.npy`
- `metadata.json`

The metadata includes:

- dataset
- representation
- model
- parameters
- normalization flag
- pipeline version
- generation timestamp
- dtype
- shape
- device used
- generation time
- text prefix
- fold and split when applicable

## Outputs

Each run writes:

- `artifacts/runs/{representation}/{dataset}/fold_metrics.csv`
- `artifacts/runs/{representation}/{dataset}/summary.json`

The summary contains:

- run status
- selected configuration
- per-fold metrics
- aggregated mean and standard deviation for:
  - `test_f1_macro`
  - `test_accuracy`
  - `test_precision_macro`
  - `test_recall_macro`

## Tests

Run the test suite with:

```bash
.venv/bin/pytest -q
```

Current tests cover:

- dataset validation
- cache hit/miss behavior
- metadata fields
- model-specific text prefixes
- split integrity
- end-to-end lexical run

## Notes

- Device selection is automatic: `cuda` when available, otherwise `cpu`
- Dense local models use `sentence-transformers`
- Remote models use the OpenRouter API through the `openai` client interface
- Heavy models may fail or be skipped depending on local hardware and installed dependencies
