<div align="center">

# VocalTwin

**Personalised voice cloning&nbsp;+&nbsp;TTS**  
Train on your own recordings – then turn any text into speech that sounds like **you**.

[![Python](https://img.shields.io/badge/python-3.9--3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Issues](https://img.shields.io/github/issues/shamspias/VocalTwin.svg?color=critical)](https://github.com/shamspias/VocalTwin/issues)

</div>

---

## Table of Contents

1. [Features](#features)  
2. [Quick Start](#quick-start)  
3. [Installation](#installation)  
4. [Usage](#usage)  
   - [Train only](#1-train-on-your-voice)  
   - [Synthesize only](#2-synthesize-text)  
   - [Both in one pass](#3-all-in-one)  
5. [Directory Layout](#directory-layout)  
6. [Troubleshooting & Tips](#troubleshooting--tips)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [Acknowledgements](#acknowledgements)

---

## Features

- **OpenVoice V2** tone-colour converter  
- **MeloTTS** multilingual base TTS (EN · ES · FR · ZH · JA · KO …)  
- Simple **CLI** – `train`, `synthesize`, `train_and_synthesize`  
- Works with **one or many** MP3 samples (more = better)  
- Pure-Python, GPU-accelerated (falls back to CPU)  
- Clean, class-based code; easy to extend

---

## Quick Start

<details>
<summary>Five-minute demo (Linux / macOS / WSL)</summary>

```bash
git clone https://github.com/shamspias/VocalTwin.git
cd VocalTwin
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unidic download                                # one-time

# download OpenVoice V2 checkpoints (~600 MB)
curl -L https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -o ckpt.zip
unzip ckpt.zip -d checkpoints_v2 && rm ckpt.zip

# drop a few clean MP3s into audio_samples/  (≈30-60 s total audio)
# create texts/hello.txt with some text

python main.py train_and_synthesize --language EN
# → outputs/hello.wav  (your voice!)
```

</details>

---

## Installation

> **Prerequisites**
>
> * Python 3.9 – 3.10
> * `ffmpeg` in your `PATH`
> * NVIDIA GPU with CUDA 11+ recommended (CPU works, just slower)

```bash
# clone + create venv
git clone https://github.com/shamspias/VocalTwin.git
cd VocalTwin
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# install deps
pip install --upgrade pip
pip install -r requirements.txt
python -m unidic download            # for Japanese tokeniser

# fetch OpenVoice V2 converter checkpoints
mkdir -p checkpoints_v2
curl -L https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip -o ckpt.zip
unzip ckpt.zip -d checkpoints_v2 && rm ckpt.zip
```

---

## Usage

### 1. Train on your voice

Put MP3s in **`audio_samples/`** (more = better):

```bash
python main.py train
```

Creates **`checkpoints/target_se.pth`** – your speaker embedding.

---

### 2. Synthesize text

Add one or more `.txt` files to **`texts/`** and run:

```bash
python main.py synthesize --language EN   # EN / ES / FR / ZH / JA / KO
```

Each `.txt` → `.wav` with your voice in **`outputs/`**.

---

### 3. All-in-one

```bash
python main.py train_and_synthesize --language EN
```

---

### CLI reference

```bash
python main.py --help
```

```
usage: VocalTwin [-h] [--audio_dir AUDIO_DIR] [--text_dir TEXT_DIR]
                 [--checkpoint_dir CHECKPOINT_DIR] [--output_dir OUTPUT_DIR]
                 [--language LANGUAGE]
                 {train,synthesize,train_and_synthesize}

Positional arguments:
  {train,synthesize,train_and_synthesize}
                        Action to perform

Optional arguments:
  --audio_dir AUDIO_DIR         MP3 training recordings           [audio_samples]
  --text_dir TEXT_DIR           Input .txt files                  [texts]
  --checkpoint_dir CHECKPOINT_DIR
                                Model checkpoints + target SE     [checkpoints]
  --output_dir OUTPUT_DIR       Generated WAVs                    [outputs]
  --language LANGUAGE           TTS language code (MeloTTS)       [EN]
```

---

## Directory Layout

```
VocalTwin/
├── audio_samples/         # your MP3 recordings
├── texts/                 # input .txt
├── outputs/               # generated WAV
├── checkpoints/           # target_se.pth lives here
├── checkpoints_v2/        # OpenVoice converter checkpoints
├── src/
│   ├── trainer.py         # extracts speaker embedding
│   ├── synthesizer.py     # TTS + tone-colour conversion
│   └── utils.py           # helpers (placeholder)
├── main.py                # CLI
└── requirements.txt
```

---

## Troubleshooting & Tips

* **Noise matters** – recordings should be clear, 16 kHz+ preferred.
* Short texts (< 3 s) sometimes clip; add punctuation or line-breaks.
* On CPU the conversion step is slow; expect \~1 × RT or worse.
* “`target_se.pth not found`”? Run the **`train`** step first or check paths.

---

## Contributing

Bug reports & PRs are welcome!

1. Fork → feature branch → PR to **`main`**
2. Follow the existing code style (black + isort).
3. Add/update docstrings and a short demo if introducing a new feature.

---

## License

`VocalTwin` is released under the **MIT License** – see [`LICENSE`](LICENSE).

---

## Acknowledgements

* **[OpenVoice V2](https://github.com/myshell-ai/OpenVoice)** – tone-colour converter
* **[MeloTTS](https://github.com/myshell-ai/MeloTTS)** – multilingual neural TTS
* Demo recordings courtesy of the open-source speech community

---

<p align="center">Made with Frustration 🫤 by <a href="https://github.com/shamspias">@shamspias</a></p>
