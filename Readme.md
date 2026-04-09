# XTTSv2 Finetuning Guide for New Languages - Maltese

This is a forked version of [anhnh2002/XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages).

This guide provides instructions for finetuning XTTSv2 on a new language, using Maltese (`mt`) as an example.

## Table of Contents
1. [Installation](#1-installation)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Data Preparation](#3-data-preparation)
4. [Pretrained Model Download](#4-pretrained-model-download)
5. [Vocabulary Extension and Configuration Adjustment](#5-vocabulary-extension-and-configuration-adjustment)
6. [DVAE Finetuning (Optional)](#6-dvae-finetuning-optional)
7. [GPT Finetuning](#7-gpt-finetuning)
8. [TensorBoard Monitoring](#8-tensorboard-monitoring)
9. [Usage Example](#9-usage-example)

## 1. Installation

Using Google Colab:

Follow these steps in the provided Colab notebook:
 
1. **Mount Google Drive** to access your datasets and save checkpoints:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
 
2. **Install Python 3.10** and set it as the default interpreter.
 
3. **Check GPU availability**:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
 
4. **Clone the repository**:
   ```bash
   git clone https://github.com/Fabzamm/XTTSv2-Finetuning-for-New-Languages.git
   cd XTTSv2-Finetuning-for-New-Languages
   ```
 
5. **Fix blinker dependency conflict** (Colab sometimes ships a conflicting version):
   ```bash
   !find /usr/lib/python3 -name "blinker*" -exec rm -rf {} + 2>/dev/null
   !find /usr/local/lib/python3.10 -name "blinker*" -exec rm -rf {} + 2>/dev/null
   ```

6. **Install TorchCodec**:
   ```bash
   !pip install torchcodec
   ```
 
7. **Install requirements**:
   ```bash
   cd /content/XTTSv2-Finetuning-for-New-Languages
   pip install -r requirements.txt
   ```

## 2. Pipeline Overview

The full finetuning pipeline follows these steps:
 
1. **Google Drive** → Load datasets (MASRI + Common Voice)
2. **Environment setup** — Python 3.10, GPU check, install libraries
3. **Clone XTTS repo** and navigate into it
4. **Download pretrained model** (XTTS v2 base checkpoint)
5. **Extend vocabulary** for Maltese using training metadata and optionally Korpus Malti
6. **Train DVAE** — teaches the model to encode audio into discrete acoustic tokens
7. **Train GPT** — finetunes the language model to map text to audio tokens
8. **Monitor with TensorBoard** — track loss and training progress
9. **Run inference** — generate speech from text using the finetuned model
 
## 3. Data Preparation

### Datasets Used for Maltese
 
Two datasets are used for Maltese finetuning:
 
- **MASRI dataset** — Maltese speech corpus
- **Common Voice dataset** — Mozilla's crowd-sourced Maltese speech data
 
Configure the paths at the top of your notebook or script:
 
```python
LANGUAGE = "mt"
MASRI_DIR = "/content/drive/MyDrive/..."
CV_DIR = "/content/drive/MyDrive/..."
```

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

## 4. Pretrained Model Download

Execute the following command to download the pretrained XTTS v2 base model:

```bash
python download_checkpoint.py --output_path checkpoints/
```

## 5. Vocabulary Extension and Configuration Adjustment

Extend the vocabulary and adjust the configuration with:

```bash
python extend_vocab_config.py \
  --output_path=checkpoints/ \
  --metadata_path datasets/metadata_train.csv \
  --language mt \
  --extended_vocab_size 1000 \
  --use_korpus \
  --korpus_max_samples 50000
```

### Flag Reference
 
| Flag | Description |
|---|---|
| `--use_korpus` | Also trains the tokenizer on the [Korpus Malti](https://huggingface.co/datasets/MLRS/korpus_malti) streaming dataset, giving broader Maltese vocabulary coverage |
| `--korpus_max_samples` | Maximum number of sentences to stream from Korpus Malti. Set to `-1` for no limit |
 
> **Note:** The `masri/` directory contains Maltese-specific tokenization logic used during vocabulary extension. In particular, `masri/tokeniser/km_tokeniser.py` implements a Maltese tokenizer that is used internally when processing the training data.
 
---

## 6. DVAE Finetuning (Optional)

The DVAE (Discrete Variational Autoencoder) learns to convert raw audio into discrete acoustic tokens. This is a compact representation the GPT model later learns to predict. Finetuning it on your target language can improve audio quality.

```bash
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=datasets/metadata_train.csv \
--eval_csv_path=datasets/metadata_eval.csv \
--language="mt" \
--num_epochs=5 \
--batch_size=128 \
--lr=5e-6
```

> **Tip:** If you have approximately 20 hours of short audio clips in your dataset, DVAE finetuning is not required — the pretrained DVAE generalises well enough.
 
---

## 7. GPT Finetuning

The GPT model is the core of XTTS. It learns to map text tokens to audio tokens conditioned on a speaker reference. Finetuning it on your target language data gives the model natural pronunciation and prosody in that language.

```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
--output_path=checkpoints/ \
--metadatas "datasets/metadata_train.csv,datasets/metadata_eval.csv,mt" \
--num_epochs=5 \
--batch_size=2 \
--grad_acumm=16 \
--max_text_length=400 \
--max_audio_length=330750 \
--weight_decay=1e-2 \
--lr=5e-6 \
--save_step=4443 \
--save_n_checkpoints=100
```

### Resuming Training from a Checkpoint
 
Use `--restore_path` to resume an interrupted training run:
 
```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
  --output_path=checkpoints/ \
  --metadatas "datasets/metadata_train.csv,datasets/metadata_eval.csv,mt" \
  --restore_path "checkpoints/GPT_XTTS_FT-.../best_model.pth" \
  --num_epochs=5 \
  ...
```

## 8. TensorBoard Monitoring

You can monitor training loss and progress in real time using TensorBoard:

```python
%tensorboard --logdir checkpoints/run/training/
```
 
This is particularly useful during GPT finetuning to detect overfitting or confirm the loss is converging as expected.
 
---

## 9. Usage Example

Here's a sample code snippet demonstrating how to use the finetuned model:

```python
import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model paths
xtts_checkpoint = "checkpoints/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/best_model_99875.pth"
xtts_config = "checkpoints/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/config.json"
xtts_vocab = "checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Model loaded successfully!")

# Inference
tts_text = "Toni tagħna tani tina talli tajtu tuta tajba."
speaker_audio_file = "ref.wav"
lang = "mt"

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

tts_texts = sent_tokenize(tts_text)

wav_chunks = []
for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))

out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

# Play audio (for Jupyter Notebook)
from IPython.display import Audio
Audio(out_wav, rate=24000)
```

## Notes
 
> The first two notes below are from the original upstream repository: [anhnh2002/XTTSv2-Finetuning-for-New-Languages](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages)
 
- **HiFiGAN decoder finetuning** was attempted but resulted in worse performance and is not recommended.
- **DVAE finetuning** is optional if you have ~20 hours of short audio clips — the pretrained DVAE generalises well.
- **GPT finetuning** is the most impactful step and is always recommended.
- The `masri/` directory contains Maltese-specific tokenization logic (including `masri/tokeniser/km_tokeniser.py`) used during vocabulary extension. Do not remove it.
