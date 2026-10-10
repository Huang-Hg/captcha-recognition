# Multi-source Text CAPTCHA Recognition

A research framework for multi-source text CAPTCHA recognition: digits, uppercase
and lowercase letters, variable lengths, distortion and occlusion.

The current model shares a two-dimensional SVTRv2 visual encoder [1, 2] between CTC
[3] and a causal Transformer attention decoder [4]. Checkpoints own their vocabulary, input geometry and
preprocessing contract. The selected research candidate uses CTC multiview decoding.
Strong cross-source generalization remains an open goal; the candidate is not a
validated universal recognizer.

## Repository contents

- `src/captcha_recognition/`: model, experimental variants, training, inference and evaluation.
- `src/captcha_recognition/backbones/`: bundled SVTR encoder and CTC components, with source attribution and license.
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
```

This installs the workspace in editable mode; use `python -m pip install .` for a
regular installation. The network definitions needed by the project checkpoints
are bundled. No external OpenOCR installation, source checkout, YAML configuration
or vocabulary file is needed for recognition or continuation training.

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

## Method attribution

The following table distinguishes implementation lineage from conceptual references.
The current hybrid is a project-specific combination, not a reproduction of any
single cited paper. Citation does not imply that the original authors evaluated
this implementation or endorse its reported results.

| Component or experiment | References | Relationship to this implementation |
| --- | --- | --- |
| Visual encoder and CTC feature neck | SVTRv2 [1], SVTR [2] | Code and pretrained encoder lineage from OpenOCR. The bundled encoder is the released `SVTRv2LNConvTwo33` server variant. This project does not reproduce the complete SVTRv2 training recipe or all of its proposed modules. |
| Sequence alignment and loss | CTC [3] | Uses PyTorch CTC loss and project decoding utilities. |
| Causal attention branch | Transformer [4] | Uses PyTorch Transformer decoder layers with project tokenization, two-dimensional positional encoding and training integration. |
| Joint CTC/attention and rescoring | Hori et al. [5] | Conceptual reference from speech recognition. The project's finite-candidate rescoring is not the paper's joint beam-search algorithm. |
| Autoregressive scene-text recognition | PARSeq [6] | Related work and design reference only; no PARSeq weights, permutation training or bidirectional refinement are used. |
| Earlier DenseNet/SE/GRU models | DenseNet [7], SE [8], GRU [9] | Architectural building blocks in the legacy `dense_ctc*.py` research models. GRU is also used by the `column_recurrent` experiment. |
| Original DFCR project direction | Wang et al. [10] | Historical CAPTCHA-recognition reference. The current SVTR hybrid and later CTC models are not the paper's fixed-position classifier or an exact reproduction. |

Foreground residual gating, color-view probability mixing and the four feature
adapters are project-specific experiments. Their names describe implementations,
not established published methods or demonstrated novel contributions. Their
limited evaluation is described under Research status above.

## Code and data provenance

The bundled encoder and CTC code is adapted from the
[OpenOCR authors and contributors](https://github.com/Topdu/OpenOCR/tree/1ccfc6ee6161f7133b192e16af3a9265273997ba),
revision `1ccfc6ee6161f7133b192e16af3a9265273997ba`. Source file mappings and local
changes are recorded in [NOTICE](src/captcha_recognition/backbones/NOTICE); the
upstream [Apache-2.0 license](src/captcha_recognition/backbones/LICENSE) is retained
alongside the code and included in installed packages. Removing the external
runtime dependency does not remove this code's origin or attribution obligations.
The checkpoint's `upstream_commit` field records provenance, not a runtime path.

Public dataset revisions,
labels and licenses are recorded in `configs/public_data_sources.json`. In particular,
DarkNight data includes a noncommercial restriction; model use must account for the
training data's terms. Published labels have not all been independently reannotated,
and overlap with upstream pretraining data is unknown.

## References

1. Yongkun Du, Zhineng Chen, Hongtao Xie, Caiyan Jia, and Yu-Gang Jiang.
   **SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition.**
   ICCV, 2025, pp. 20147–20156.
   [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Du_SVTRv2_CTC_Beats_Encoder-Decoder_Models_in_Scene_Text_Recognition_ICCV_2025_paper.html).

2. Yongkun Du, Zhineng Chen, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li,
   Yuning Du, and Yu-Gang Jiang.
   **SVTR: Scene Text Recognition with a Single Visual Model.**
   IJCAI, 2022, pp. 884–890.
   [Paper and DOI: 10.24963/ijcai.2022/124](https://www.ijcai.org/proceedings/2022/124).

3. Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber.
   **Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.**
   ICML, 2006, pp. 369–376.
   [Author's paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf);
   [DOI: 10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891).

4. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
   Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin.
   **Attention Is All You Need.**
   Advances in Neural Information Processing Systems 30, 2017.
   [Paper](https://arxiv.org/abs/1706.03762).

5. Takaaki Hori, Shinji Watanabe, and John Hershey.
   **Joint CTC/attention decoding for end-to-end speech recognition.**
   ACL, 2017, pp. 518–529.
   [Paper and DOI: 10.18653/v1/P17-1048](https://aclanthology.org/P17-1048/).

6. Darwin Bautista and Rowel Atienza.
   **Scene Text Recognition with Permuted Autoregressive Sequence Models.**
   ECCV, 2022.
   [Paper](https://arxiv.org/abs/2207.06966).

7. Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger.
   **Densely Connected Convolutional Networks.**
   CVPR, 2017, pp. 4700–4708.
   [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html).

8. Jie Hu, Li Shen, and Gang Sun.
   **Squeeze-and-Excitation Networks.**
   CVPR, 2018, pp. 7132–7141.
   [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html).

9. Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau,
   Fethi Bougares, Holger Schwenk, and Yoshua Bengio.
   **Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.**
   EMNLP, 2014, pp. 1724–1734.
   [Paper and DOI: 10.3115/v1/D14-1179](https://aclanthology.org/D14-1179/).

10. Jing Wang, Jiaohua Qin, Xuyu Xiang, Yun Tan, and Nan Pan.
    **CAPTCHA recognition based on deep convolutional neural network.**
    Mathematical Biosciences and Engineering, 2019, 16(5), pp. 5851–5861.
    [Publisher's paper](https://www.aimspress.com/fileOther/PDF/MBE/mbe-16-05-292.pdf);
    [DOI: 10.3934/mbe.2019292](https://doi.org/10.3934/mbe.2019292).

When describing or reusing this model, cite the methods actually used, retain the
bundled code's notices, and identify the project revision and checkpoint separately.
Reported scores above are this project's observed diagnostics, not results from
the cited papers or a claim of matching their benchmarks.
