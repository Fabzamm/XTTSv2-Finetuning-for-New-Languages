# XTTSv2 Finetuning Guide for New Languages - Maltese

This is a forked version of [anhnh2002/XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages).

This guide provides instructions for finetuning XTTSv2 on a new language, using Maltese (`mt`) as an example.

## Table of Contents
1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Preparation](#2-data-preparation)
3. [Fine-Tuning](#3-fine-tuning)
4. [Inference](#4-inference)
5. [Notes](#5-notes)

## 1. Pipeline Overview

The full finetuning pipeline follows these steps:
 
1. **Prepare datasets** — organise the MASRI and Common Voice datasets.
2. **Open the fine-tuning notebook** — use `finetuning_xtts.ipynb` in Google Colab.
3. **Set up the environment** — mount Google Drive, install Python 3.10, check GPU availability, clone the repository, and install dependencies.
4. **Configure dataset paths** — set the paths for the MASRI dataset, Common Voice dataset, combined metadata files, and checkpoints.
5. **Download the pretrained XTTSv2 model** — download the base XTTSv2 checkpoint.
6. **Extend vocabulary** — adapt the tokenizer for Maltese using the training metadata and optionally Korpus Malti.
7. **Train DVAE** — optionally finetune the DVAE component on Maltese audio.
8. **Train GPT** — finetune the main XTTSv2 GPT model.
9. **Monitor training** — optionally use TensorBoard to track training loss and progress.
10. **Run inference** — use `inference_xtts.ipynb` to generate Maltese speech using the finetuned model.
 
## 2. Data Preparation

### Datasets Used for Maltese
 
Two datasets are used for Maltese finetuning:
 
- **MASRI dataset** — Maltese speech corpus
- **Common Voice dataset** — Mozilla's crowd-sourced Maltese speech data

### Directory Structure

Ensure your data is organised as follows:

```
project_root/
├── datasets-1/
│   ├── wavs/
│   │   ├── xxx.wav
│   │   ├── yyy.wav
│   │   ├── zzz.wav
│   │   └── ...
│   ├── metadata_train.csv
│   ├── metadata_eval.csv
├── datasets-2/
│   ├── wavs/
│   │   ├── xxx.wav
│   │   ├── yyy.wav
│   │   ├── zzz.wav
│   │   └── ...
│   ├── metadata_train.csv
│   ├── metadata_eval.csv
...
│
├── masri/
├── recipes/
├── scripts/
├── TTS/
└── README.md
```

Format your `metadata_train.csv` and `metadata_eval.csv` files as follows:

```
audio_file|text|speaker_name
wavs/xxx.wav|How do you do?|@X
wavs/yyy.wav|Nice to meet you.|@Y
wavs/zzz.wav|Good to see you.|@Z
```

For Maltese training, the notebook expects paths for the MASRI dataset, Common Voice dataset, and the combined training and evaluation metadata files.

Example:

```bash
LANGUAGE = "mt"

MASRI_DIR = "/content/drive/MyDrive/path/to/MASRI"
CV_DIR = "/content/drive/MyDrive/path/to/CV"

COMBINED_DIR = "/content/drive/MyDrive/path/to/Combined"
CHECKPOINT_DIR = "/content/drive/MyDrive/path/to/checkpoints"
```

## 3. Fine-Tuning

Fine-tuning can be performed using the provided Google Colab notebook:

```text
finetuning_xtts.ipynb
```

This notebook prepares the Colab environment, loads the datasets, downloads the pretrained XTTSv2 model, extends the vocabulary for Maltese, optionally trains the DVAE component, trains the GPT component, and optionally opens TensorBoard for monitoring.

### Fine-Tuning Notebook Steps

Open `finetuning_xtts.ipynb` in Google Colab and follow these steps.

---

### 1. Mount Google Drive

The notebook first mounts Google Drive so that datasets, checkpoints, and training outputs can be accessed and saved.

```python
from google.colab import drive
drive.mount('/content/drive')
```

This is important because Colab runtime storage is temporary. Saving checkpoints to Google Drive prevents them from being lost when the session disconnects.

---

### 2. Set Up the Colab Environment

The notebook installs Python 3.10, sets it as the default interpreter, reinstalls `pip`, and verifies the Python version.

It also checks whether a GPU is available:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

A GPU is strongly recommended for XTTSv2 fine-tuning.

---

### 3. Clone or Open the Repository

The notebook checks whether the repository already exists in the Colab runtime. If it does not exist, it clones the repository from GitHub.

```python
REPO_DIR = "/content/XTTSv2-Finetuning-for-New-Languages"
```

If the folder does not already exist, the notebook clones the repository:

```bash
git clone https://github.com/Fabzamm/XTTSv2-Finetuning-for-New-Languages.git
```

Then it changes directory into the repository folder:

```python
%cd /content/XTTSv2-Finetuning-for-New-Languages
```

---

### 4. Install Required Dependencies

The notebook installs the required packages for XTTSv2 fine-tuning.

It first removes possible conflicting `blinker` versions from Colab:

```bash
!find /usr/lib/python3 -name "blinker*" -exec rm -rf {} + 2>/dev/null
!find /usr/local/lib/python3.10 -name "blinker*" -exec rm -rf {} + 2>/dev/null
```

Then it installs TorchCodec:

```bash
!pip install torchcodec
```

Finally, it installs the repository requirements:

```bash
%cd /content/XTTSv2-Finetuning-for-New-Languages
!pip install -r requirements.txt
```

---

### 5. Configure Dataset and Checkpoint Paths

The notebook then sets the main paths used during training.

You should update these paths according to where your own datasets and checkpoints are stored in Google Drive.

```python
import pandas as pd
import os

LANGUAGE = "mt"

MASRI_DIR = "/content/drive/MyDrive/path/to/MASRI"
CV_DIR = "/content/drive/MyDrive/path/to/CV"

MASRI_TRAIN_CSV = os.path.join(MASRI_DIR, "metadata_train.csv")
MASRI_EVAL_CSV = os.path.join(MASRI_DIR, "metadata_eval.csv")

CV_TRAIN_CSV = os.path.join(CV_DIR, "metadata_train.csv")
CV_EVAL_CSV = os.path.join(CV_DIR, "metadata_eval.csv")

COMBINED_DIR = "/content/drive/MyDrive/path/to/Combined"

COMBINED_TRAIN_CSV = os.path.join(COMBINED_DIR, "metadata_train.csv")
COMBINED_EVAL_CSV = os.path.join(COMBINED_DIR, "metadata_eval.csv")

CHECKPOINT_DIR = "/content/drive/MyDrive/path/to/checkpoints"

EXTENDED_VOCAB_SIZE = 1000

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

The notebook also prints the number of rows in the combined training and evaluation metadata files. This helps confirm that the CSV files are being loaded correctly.

---

### 6. Download the Pretrained XTTSv2 Model

The notebook includes a cell to download the pretrained XTTSv2 base checkpoint:

```bash
!python download_checkpoint.py --output_path {CHECKPOINT_DIR}
```

This only needs to be run once.

If the pretrained model files already exist in your checkpoint folder, you do not need to run this cell again.

---

### 7. Extend the Vocabulary for Maltese

The notebook includes a cell to extend the tokenizer vocabulary for Maltese:

```bash
!python extend_vocab_config.py \
    --output_path={CHECKPOINT_DIR} \
    --metadata_path={COMBINED_TRAIN_CSV} \
    --language={LANGUAGE} \
    --extended_vocab_size={EXTENDED_VOCAB_SIZE} \
    --use_korpus
```

This step adapts the XTTSv2 vocabulary to better support Maltese text.

The `--use_korpus` flag also uses the [Korpus Malti](https://huggingface.co/datasets/MLRS/korpus_malti) streaming dataset to improve Maltese vocabulary coverage.

### Flag Reference

| Flag | Description |
|---|---|
| `--use_korpus` | Also trains the tokenizer on the Korpus Malti streaming dataset, giving broader Maltese vocabulary coverage |
| `--korpus_max_samples` | Maximum number of sentences to stream from Korpus Malti. Set to `-1` for no limit |
| `--extended_vocab_size` | Number of new vocabulary tokens to add for the target language |

> **Note:** The `masri/` directory contains Maltese-specific tokenization logic used during vocabulary extension. In particular, `masri/tokeniser/km_tokeniser.py` implements a Maltese tokenizer that is used internally when processing the training data.

---

### 8. Train the DVAE Component Optional

The DVAE, or Discrete Variational Autoencoder, learns to convert raw audio into discrete acoustic tokens. These tokens are later used by the GPT component.

The notebook includes a DVAE training cell, but this step is optional.

```python
DVAE_EPOCHS = 5
DVAE_BATCH_SIZE = 128
DVAE_LR = 5e-6
```

The training command is:

```bash
!python train_dvae_xtts.py \
    --output_path={CHECKPOINT_DIR} \
    --train_csv_path={COMBINED_TRAIN_CSV} \
    --eval_csv_path={COMBINED_EVAL_CSV} \
    --language={LANGUAGE} \
    --num_epochs={DVAE_EPOCHS} \
    --batch_size={DVAE_BATCH_SIZE} \
    --lr={DVAE_LR}
```

> **Tip:** If you have approximately 20 hours of short audio clips in your dataset, DVAE finetuning is not required. The pretrained DVAE usually generalises well enough.

---

### 9. Train the GPT Component

The GPT model is the main component of XTTSv2 fine-tuning. It learns to map text tokens to audio tokens conditioned on a speaker reference.

The notebook defines the GPT training hyperparameters:

```python
GPT_EPOCHS = 3
GPT_BATCH_SIZE = 2
GPT_GRAD_ACCUM = 16
GPT_MAX_TEXT_LEN = 400
GPT_MAX_AUDIO_LEN = 330750
GPT_WEIGHT_DECAY = 1e-2
GPT_LR = 5e-6
GPT_SAVE_STEP = 4443
GPT_SAVE_N_CHECKPOINTS = 100
```

The notebook supports training with both MASRI and Common Voice metadata:

```python
METADATA_ARG_MASRI = f"{MASRI_TRAIN_CSV},{MASRI_EVAL_CSV},{LANGUAGE}"
METADATA_ARG_CV = f"{CV_TRAIN_CSV},{CV_EVAL_CSV},{LANGUAGE}"
```

The GPT training command is:

```bash
!python train_gpt_xtts.py \
    --output_path {CHECKPOINT_DIR} \
    --metadatas "{METADATA_ARG_MASRI}" "{METADATA_ARG_CV}" \
    --num_epochs {GPT_EPOCHS} \
    --batch_size {GPT_BATCH_SIZE} \
    --grad_acumm {GPT_GRAD_ACCUM} \
    --max_text_length {GPT_MAX_TEXT_LEN} \
    --max_audio_length {GPT_MAX_AUDIO_LEN} \
    --weight_decay {GPT_WEIGHT_DECAY} \
    --lr {GPT_LR} \
    --save_step {GPT_SAVE_STEP} \
    --save_n_checkpoints {GPT_SAVE_N_CHECKPOINTS}
```

---

### 10. Resume GPT Training from a Checkpoint

The notebook also supports resuming training from an existing checkpoint.

To resume training, set `RESTORE_CHECKPOINT` to the path of the checkpoint you want to continue from:

```python
RESTORE_CHECKPOINT = "/content/drive/MyDrive/path/to/checkpoint.pth"
```

If you do not want to resume from a checkpoint, set:

```python
RESTORE_CHECKPOINT = None
```

The notebook automatically adds the `--restore_path` argument when a restore checkpoint is provided.

---

### 11. Monitor Training with TensorBoard

The notebook includes an optional TensorBoard cell.

TensorBoard can be used to monitor training loss and progress:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/path/to/checkpoints
```

This is useful during GPT fine-tuning to check whether the loss is decreasing and whether the model is training as expected.

---

## 4. Inference

After finetuning the XTTSv2 model, inference can be performed using the provided Colab notebook:

```text
inference_xtts.ipynb
```

This notebook loads the finetuned XTTSv2 model, uses a speaker reference audio file, generates Maltese speech from input text, and saves the generated audio as a `.wav` file.

### Inference Notebook Steps

Open `inference_xtts.ipynb` in Google Colab and follow these steps.

### 1. Mount Google Drive

The notebook first mounts Google Drive so that the checkpoint files, vocabulary file, speaker reference audio, and output folder can be accessed.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Set Up the Colab Environment

The notebook installs and configures the required environment for XTTSv2 inference.

This includes:

* installing Python 3.10
* checking the active Python version
* installing the required dependencies
* fixing possible Colab dependency conflicts

### 3. Clone or Open the Repository

The notebook checks whether the repository already exists in the Colab runtime. If it does not exist, it clones the repository from GitHub.

```python
REPO_DIR = "/content/XTTSv2-Finetuning-for-New-Languages"
```

After this, the notebook changes directory into the repository folder so that the inference script can be used.

### 4. Install Required Dependencies

The notebook installs the packages required to run inference.

This includes:

```bash
pip install torchcodec
pip install -r requirements.txt
pip install ipython
```

The notebook may also remove conflicting `blinker` versions, since Google Colab sometimes includes a version that conflicts with the required dependencies.

### 5. Configure the Inference Settings

Before running inference, update the main inference variables in the notebook.

These variables define the text to synthesise, the model files to use, the speaker reference audio, and the output location.

| Variable     | Description                                               |
| ------------ | --------------------------------------------------------- |
| `TEXT`       | The Maltese text that will be converted into speech       |
| `CHECKPOINT` | Path to the finetuned XTTSv2 checkpoint                   |
| `CONFIG`     | Path to the `config.json` file from the same training run |
| `VOCAB`      | Path to the `vocab.json` file used by the model           |
| `SPEAKER`    | Path to the reference speaker audio file                  |
| `OUTPUT`     | Path where the generated `.wav` file will be saved        |
| `SPEED`      | Speech speed. `1` means normal speed                      |

Example:

```python
TEXT = "Din hija sentenza biex nara kif jaħdem il-mudell."

CHECKPOINT = "/content/drive/MyDrive/checkpoints/GPT_XTTS_FT-.../best_model.pth"
CONFIG = "/content/drive/MyDrive/checkpoints/GPT_XTTS_FT-.../config.json"
VOCAB = "/content/drive/MyDrive/checkpoints/XTTS_v2.0_original_model_files/vocab.json"

SPEAKER = "/content/drive/MyDrive/voices/ref.wav"
OUTPUT = "/content/drive/MyDrive/outputs/generated_maltese_audio.wav"

SPEED = 1
```

> **Important:** The checkpoint, config file, and vocabulary file should belong to the same training setup. Mixing files from different runs may cause errors or poor-quality output.

### 6. Run Inference

After updating the paths and text, run the inference cell in the notebook.

The notebook uses `run_inference.py` to generate the audio:

```bash
python run_inference.py \
  --checkpoint CHECKPOINT \
  --config CONFIG \
  --vocab VOCAB \
  --speaker_audio SPEAKER \
  --text TEXT \
  --output_path OUTPUT \
  --speed SPEED
```

The generated audio will be saved to the file path specified in `OUTPUT`.

### 7. Play the Generated Audio

After inference finishes, the generated `.wav` file can be played directly inside the Colab notebook:

```python
from IPython.display import Audio

Audio(OUTPUT)
```

### Notes

The speaker reference audio should be a clear `.wav` file with minimal background noise, since XTTSv2 uses it for voice cloning.

For best results, use short and clear Maltese sentences when testing the model. Long paragraphs can be split into smaller sentences before inference.




## 5. Notes
 
> The first two notes below are from the original upstream repository: [anhnh2002/XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages)
 
- **HiFiGAN decoder finetuning** was attempted but resulted in worse performance and is not recommended.
- **DVAE finetuning** is optional if you have ~20 hours of short audio clips — the pretrained DVAE generalises well.
- **GPT finetuning** is the most impactful step and is always recommended.
- The `masri/` directory contains Maltese-specific tokenization logic (including `masri/tokeniser/km_tokeniser.py`) used during vocabulary extension. Do not remove it.
