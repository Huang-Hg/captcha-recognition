# Multi-source Text CAPTCHA Recognition

A research framework for multi-source text CAPTCHA recognition: digits, uppercase
and lowercase letters, variable lengths, distortion and occlusion.

The current model shares a two-dimensional SVTR visual encoder between CTC and a
causal attention decoder. Checkpoints own their vocabulary, input geometry and
preprocessing contract. The selected research candidate uses CTC multiview decoding.
Strong cross-source generalization remains an open goal; the candidate is not a
validated universal recognizer.

## Repository contents

- `src/captcha_recognition/`: model, experimental variants, training, inference and evaluation.
- `src/dense_ctc*.py`: legacy DenseNet/GRU models required for checkpoint compatibility.
- Other `src/` modules: SVTR adapters, preprocessing, decoding and sampling utilities.
- `configs/`: architecture contract, training example, candidate metadata and data provenance.

Local experiment scripts, datasets, weights, logs, downloaded dependencies and machine
configuration are deliberately excluded from Git. The installed commands below do
not require a `scripts/` directory. Checkpoint architecture identifiers describe the serialized model formats. Python
module filenames use descriptive English names. The package is `captcha_recognition` and public commands use the
`captcha-*` prefix.

## Setup

Use Python 3.10 or newer and install into a project virtual environment. CUDA is
required for training; inference supports CPU and CUDA. Install a PyTorch build
appropriate for your system before installing this project when necessary.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
git clone https://github.com/Topdu/OpenOCR.git assets/upstream/OpenOCR
git -C assets/upstream/OpenOCR checkout 1ccfc6ee6161f7133b192e16af3a9265273997ba
```

This is an editable workspace installation. Keep the pinned OpenOCR source checkout
at the path above; it is an external runtime dependency and is not bundled here.

Weights are not included. Place a compatible project checkpoint at the path specified
in `configs/candidate.json`, or use `--checkpoint` with an explicit file. The candidate
file records its expected SHA-256. Raw upstream OpenOCR weights cannot be substituted
for an adapted project checkpoint. Training currently requires an existing hybrid or
audited foreground-SVTR project checkpoint; a fresh checkout alone is not sufficient
to reproduce the trained candidate.

## Recognition

Run commands from the repository directory:

```bash
captcha-recognize --candidate configs/candidate.json --image data/example.png
captcha-recognize --checkpoint checkpoints/model.pth --mode ctc_multiview --image data/example.png
```

Use `--device cpu` to select CPU inference. Output is JSON. Candidate loading verifies
the checkpoint hash and metadata. Image aspect ratios must fit the checkpoint canvas;
increasing inference dimensions alone does not adapt a trained model. Sequence scores
are diagnostics, not calibrated confidence.

## Training and evaluation

```bash
captcha-train --config configs/training.json --initial-checkpoint checkpoints/initial.pth --data data/dataset --output logs/new_experiment
captcha-evaluate --checkpoint checkpoints/model.pth --mode ctc_multiview --manifest data/dataset/configs/evaluation_manifest.json --output logs/evaluation.json
```

`configs/training.json` is a two-stage training example. Use `--ctc-control` for a
matched continuation without attention training. Experimental feature adapters
(`pixel_detail`, `column_recurrent`, `spatial_experts`, `column_rectifier`) require
`variant` in the configuration, `training.variant_lr`, a compatible hybrid parent
checkpoint and `--ctc-control`. They share an encoder; they are not independent
pretrained backbones or claimed novel research contributions.

The dataset root must contain `configs/train_manifest.json`,
`configs/validation_manifest.json` and `configs/evaluation_manifest.json`.
Each contains a nonempty `samples` array. A sample has `path`, `label`, `group` and
`sha256`; optional foreground annotations add `mask_path` and `mask_sha256`.
Use absolute image paths or paths relative to the command's working directory.
Hashes are SHA-256 of file bytes. Answers and image hashes must be disjoint across
splits. Evaluation samples are audited for split isolation but not used to train or
select weights. Training caches images on the GPU, so dataset size affects memory.

Model selection uses a balanced validation score: half the public-source macro
exact accuracy and half the synthetic-group macro exact accuracy. See
`src/validation_metrics.py` for group classification. Keep evaluated public samples
marked as observed diagnostics; repeated evaluation is not a fresh blind test.

## Research status

The retained candidate uses a 64x448 input and a case-sensitive 62-character alphabet.
Its balanced validation score is **91.18%**, not a universal accuracy estimate.
Observed held-out-source exact accuracy remains **329/500 (65.8%)** on Russian market
and **59/168 (35.1%)** on the in-scope ChangAn subset. Generalization is unresolved.

Four feature-adapter experiments and a matched CTC continuation used one 600-step run
per condition. None surpassed the retained candidate under the final multiview
validation criterion. This conclusion is limited to the tested initialization and
budget. The repository does not include the local experiment logs or model binaries.

## Dependencies and data provenance

The encoder integration depends on [OpenOCR](https://github.com/Topdu/OpenOCR), pinned
above and distributed under its own Apache-2.0 license. Public dataset revisions,
labels and licenses are recorded in `configs/public_data_sources.json`. In particular,
DarkNight data includes a noncommercial restriction; model use must account for the
training data's terms. Published labels have not all been independently reannotated,
and overlap with upstream pretraining data is unknown.
