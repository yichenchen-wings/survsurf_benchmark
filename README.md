
# SurvSurf benchmark

This repository contains the data adapters, PyTorch Lightning training code,
model wrappers, evaluation utilities, and notebooks used to benchmark SurvSurf
against DeepHit and conventional survival-analysis baselines. The task is to
estimate the probability that an ordered event or severity grade has occurred by
time `t`, conditional on subject-level features.

The repository is research code. Experiments are configured with JSON files and
logged to Weights & Biases (W&B); data preparation and result aggregation are
primarily performed in notebooks.

## Repository layout

| Path | Purpose |
| --- | --- |
| `train_SurvSurf.py` | Train SurvSurf from a JSON configuration. |
| `train_DeepHit.py` | Train DeepHit from a JSON configuration. |
| `train_CoxTDNN.py` | Legacy fixed-configuration Cox time-dependent NN run. |
| `model_factory_*.py` | Model constructors, objectives, and Lightning wrappers. |
| `dataset_*.py` | Dataset-specific CSV-to-tensor transformations and data modules. |
| `datasets.py` | Data-module registry used by the training entry points. |
| `model_eval_utils.py` | IPCW Brier/AUC and error/monotonicity metrics. |
| `pl_train.py`, `pl_wrapper.py` | Shared Lightning training loop and wrapper. |
| `torch_deephit/` | Local PyTorch implementation of DeepHit. |
| `wandb_config_*.json` | Reproducible experiment configurations. |
| `notebook_*.ipynb` | Preparation, baseline fitting, evaluation, and figures. |
| `dataset_split/` | Prepared train/tune/validation/test CSV files. |
| `env_spec_python/` | Conda and pip environment specifications. |

## Installation

Conda or Mamba is recommended because the environment includes compiled survival
analysis dependencies.

```bash
conda env create -f env_spec_python/environment.yml
conda activate env_survsurf_benchmark
python -m pip install -r env_spec_python/pip_requirements.txt
```

Confirm that `python` and `pip` resolve to the new environment with `which python`
and `which pip`. The pip requirements include the SurvSurf package. If that
dependency points to a private repository, installation requires a Git access
token; contact the project maintainers for access. Training also requires a W&B
account because the entry points call `wandb.login()`.

## Data format

Each dataset is represented by split-specific CSV files. Given dataset name
`<dataset>` and split `<split>` (`train`, `tune`, `val`, or `test`), adapters look
for files such as:

```text
dataset_split/<dataset>__df_features_<split>.csv
dataset_split/<dataset>__df_state_history_sampled_max_<split>.csv
dataset_split/<dataset>__df_true_prob_long_<split>.csv  # simulated data only
```

Feature files contain one row per subject, a `subject` identifier, and columns
named `feat...`. State-history files contain at least `subject`, `t`, and
`g_max_by_time`. Adapters convert trajectories to survival rows:

| Field | Meaning |
| --- | --- |
| `subject` | Subject or trajectory identifier. |
| `X` | Subject features; for DeepHit this also includes grade. |
| `g` | Normalized grade, returned separately for SurvSurf. |
| `t` | Event or censoring time. |
| `y` | Event-observed indicator. |
| `weight` | Per-row training weight. |
| `is_trans` | Row is at or near an observed transition. |

`separate_g_from_feats=True` produces the seven-item SurvSurf batch
`(subject, X, g, t, y, weight, is_trans)`. When false, grade is appended to `X`
and the batch is `(subject, X, t, y, weight, is_trans)`.

The `mode`/`train_mode` setting controls trajectory conversion. Common modes are:

- `first_cross_obs_only`: first crossings plus a censored (immediately) higher grade.
- `first_cross_obs_only_more_g`: first crossings plus multiple censored higher grades.
- `full_traj_obs_only`: labels at all observed trajectory times.
- `first_last_obs_per_g`: first and last informative observation per grade.
- `all_tg`: every observed time/grade pair (Markov adapter).
- `true_probs_grid`: known simulated ground-truth probabilities on a grid (specific to Markov datasets).
- `true_probs_grid_naless`: evaluation grid with unavailable values removed.
- `multi_t`: an evaluation grid constructed at multiple time points.

Not every adapter implements every mode; its constructor's `Literal` annotation
lists the supported values.

## Running an experiment

Activate the environment, authenticate with W&B, and pass a seed and checked-in
configuration:

```bash
python train_SurvSurf.py \
  --seed 10 \
  --config wandb_config_SurvSurf2DTaddTG_markov_censored_LossDyDg.json

python train_DeepHit.py \
  --seed 10 \
  --config wandb_config_deephit_markov_censored_LossSumo.json
```

The `train_script_*_local.sh` files show how the study runs five seeds (`10` to
`50`). Select the active configuration in those scripts before launching them.
The Python entry points currently contain a dataset path specific to the original
checkout; update their `df_dir` argument if the project is located elsewhere.

### Configuration reference

The entry points resolve `datamodule`, `model_getter`, and `loss` by name from
Python modules. Misspelled names therefore fail during startup.

| Key | Description |
| --- | --- |
| `project_name` | W&B project name. |
| `dir_runtime_results` | Lightning logs and checkpoint directory. |
| `ds_name`, `datamodule` | Dataset prefix and registered data-module class. |
| `train_mode`, `eval_mode` | Row-generation modes for fitting/evaluation. |
| `model_getter`, `loss` | Factory function and objective class names. |
| `g_resol`, `t_res_at_trans` | Grade and trajectory-time resolutions. |
| `t_max`, `t_res_in_loss` | Model time range and loss integration resolution. |
| `n_hidden_layers`, `n_hidden_dim`, `dropout` | Network architecture. |
| `batch_size`, `lr`, `weight_decay` | Optimization settings. |
| `max_epoch`, `patience` | Training limit and early-stopping patience. |
| `accum_grad_batches` | Batches over which gradients accumulate. |
| `device` | Lightning accelerator (`cpu`, `gpu`, or `auto`). |
| `save_top_k` | Best validation-loss checkpoints to retain. |
| `watch_model` | Whether W&B records gradients and parameters. |

## Reproducing the benchmark (please move to the `with_notebooks_and_data` branch)

1. Run the data-wrangling notebooks. NCT00981058 must be obtained separately from
   Data Sphere because access is controlled.
2. Run the matching `notebook_train_test_split_*.ipynb` notebook.
3. Train SurvSurf and DeepHit with the desired configurations and seeds.
4. Run the `notebook_fit_sksurv_*`, `notebook_fit_cox_time_*`, and/or
   `notebook_fit_xgboost_*` notebooks for baseline models.
5. Run the matching `notebook_Check_val_set_*_ipcw_certain.ipynb` notebooks for
   held-out metrics. They can also be changed to target the validation set.
6. Run `notebook_present_model_eval_metrics_by_dataset.ipynb` and, if needed, its
   supplement notebook to create tables and figures in `output_fig/`.

With prepared data and trained neural checkpoints, only steps 4–6 are required.
Omit clinical-trial sections if the restricted NCT00981058 data is unavailable.

## Extending the benchmark

To add a dataset, follow an existing `dataset_*.py` adapter, register its data
module in `datasets.py`, and add its feature count to the adapter's
`ds_name_to_n_feats_mapping`. Preserve the batch contracts described above.

The primary SurvSurf wrapper is `SurvSurf2DTaddTGNormTG` in
`model_factory_survsurf.py`. It can be extended with an encoder (for example, a
CNN for image input) that transforms `xs` before calling the parent model's
`forward` implementation.

## Notes and limitations

- Several notebooks and the CoxTDNN entry point contain experiment-specific paths
  or settings and may need local edits.
- Evaluation expects ordered grades and specific pandas column names; assertions
  intentionally fail early when those assumptions are violated.
- Outputs and artifacts are tracked through W&B and `runtime_results` rather than
  a standalone command-line report.


