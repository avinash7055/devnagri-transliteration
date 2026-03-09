# 🔤 Devnagri — Multilingual Transliteration Model

A character-level Transformer model that transliterates English/Romanized text into **Hindi** (हिन्दी), **Bengali** (বাংলা), and **Tamil** (தமிழ்) scripts. Built with [OpenNMT-py](https://opennmt.net/), optimized with [CTranslate2](https://github.com/OpenNMT/CTranslate2) for fast CPU inference, and deployed as an interactive [Gradio](https://gradio.app/) demo.

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training Pipeline](#-training-pipeline)
- [Evaluation Results](#-evaluation-results)
- [Sample Outputs](#-sample-outputs)
- [CTranslate2 Optimization](#-ctranslate2-optimization)
- [Deployment](#-deployment)
- [Architecture &amp; Design](#-architecture--design)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

- **3 Indic Languages**: Hindi (हिन्दी), Bengali (বাংলা), Tamil (தமிழ்)
- **Single Multilingual Model**: One model handles all 3 languages via `<hin>`, `<ben>`, `<tam>` prefix tokens
- **7.5M Parameters**: Compact Transformer (4 layers, 256 hidden, 4 heads)
- **CTranslate2 Optimized**: int8 quantized — 12 MB model, runs on any CPU
- **Interactive Demo**: Gradio-based UI for real-time transliteration
- **Trained on Aksharantar**: 300K pairs (100K per language) from AI4Bharat's dataset

---

## 🎮 Demo

**🌐 Live Demo**: [https://huggingface.co/spaces/avi705/devnagri-transliteration](https://huggingface.co/spaces/avi705/devnagri-transliteration)

Run the Gradio demo locally:

```bash
python deploy/app.py --model_dir models/ct2_model
# Opens at http://localhost:7860
```

Type any English/Romanized word and instantly see transliterations in all 3 scripts!

---

## 📁 Project Structure

```
devnagri/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore
│
├── notebooks/
│   ├── train_colab.ipynb               # ⭐ Google Colab training notebook
│   └── evaluate_and_export.ipynb       # Post-training evaluation & export
│
├── data/
│   ├── download_data.py                # Download Aksharantar from HuggingFace
│   ├── preprocess.py                   # JSONL → character-level parallel text
│   └── processed/                      # Preprocessed train/val/test files
│
├── configs/
│   └── transliteration.yaml            # OpenNMT-py training configuration
│
├── train/
│   ├── train.py                        # Training launcher script
│   └── evaluate.py                     # Evaluation (Accuracy, CER, WER)
│
├── models/
│   └── ct2_model/                      # ⚡ CTranslate2 optimized model (int8)
│       ├── config.json
│       ├── model.bin                   # 12 MB quantized model
│       └── shared_vocabulary.json
│
├── results/
│   ├── eval_results.json               # Per-language evaluation metrics
│   └── predictions.txt                 # Test set predictions (15K pairs)
│
├── optimize/
│   ├── convert_ct2.py                  # OpenNMT → CTranslate2 conversion
│   └── benchmark.py                    # Speed/size/quality benchmarking
│
├── deploy/
│   ├── app.py                          # Gradio demo application (local)
│   ├── app_hf.py                       # Hugging Face Spaces version
│   └── requirements.txt                # HF Spaces requirements
│
└── docs/
    └── report.md                       # Technical report
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google account (for Colab training)

### Install & Run

```bash
# Clone the repo
git clone https://github.com/avinash7055/devnagri-transliteration.git
cd devnagri-transliteration

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install "numpy<2"        # Required for torch/ctranslate2 compatibility

# Run the demo (if model already exists)
python deploy/app.py --model_dir models/ct2_model
```

---

## 🔧 Training Pipeline

### Step 1: Download Data

```bash
python data/download_data.py
```

Downloads Hindi, Bengali, and Tamil subsets from the [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) dataset via HuggingFace Hub.

| Language | Raw Pairs Available |
| -------- | ------------------- |
| Hindi    | 1,299,155           |
| Bengali  | 1,231,428           |
| Tamil    | ~1,000,000+         |

### Step 2: Preprocess Data

```bash
python data/preprocess.py
```

Converts JSONL to character-level parallel text with language prefix tokens:

- **Input** (src): `<hin> n a m a s t e`
- **Output** (tgt): `न म स ् त े`

| Split | Pairs per Language | Total   |
| ----- | ------------------ | ------- |
| Train | 100,000            | 300,000 |
| Val   | 5,000              | 15,000  |
| Test  | 5,000              | 15,000  |

### Step 3: Train on Google Colab

Since training requires a GPU, use the provided Colab notebook:

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Mount Google Drive (checkpoints are saved there for safety)
3. Upload the preprocessed `data/processed/` files
4. Run all cells (~3 hours on T4 GPU)
5. Checkpoints are saved every 5,000 steps to Google Drive

> ⚠️ **Important Fix**: OpenNMT-py 3.5.1 requires `model_type: text` (not `transformer`) and `numpy<2`. The notebook handles both automatically.

**Training Configuration:**

| Parameter              | Value                              |
| ---------------------- | ---------------------------------- |
| Architecture           | Transformer (encoder-decoder)      |
| Model Type             | `text` (OpenNMT-py 3.5.x)        |
| Encoder/Decoder Layers | 4 + 4                              |
| Hidden Size            | 256                                |
| Attention Heads        | 4                                  |
| Feed-Forward Size      | 1024                               |
| Total Parameters       | **7,458,000**                |
| Optimizer              | Adam (Noam LR schedule)            |
| Base Learning Rate     | 2.0 (peak ~0.002 at step 4000)     |
| Warmup Steps           | 4,000                              |
| Batch Size             | 4,096 tokens                       |
| Gradient Accumulation  | 4 steps (effective: 16,384 tokens) |
| Training Steps         | 30,000                             |
| Label Smoothing        | 0.1                                |
| Dropout                | 0.1                                |
| Vocabulary             | Shared, 208 tokens                 |

**Training Progress:**

| Step   | Train Acc | Train PPL | Val Acc          | Val PPL        |
| ------ | --------- | --------- | ---------------- | -------------- |
| 1,000  | 69.7%     | 6.1       | 74.25%           | 5.26           |
| 2,000  | 88.2%     | 3.3       | 85.84%           | 3.64           |
| 3,000  | 90.0%     | 3.1       | 87.50%           | 3.43           |
| 5,000  | 93.0%     | 2.8       | 89.62%           | 3.24           |
| 30,000 | 92.91%    | 2.94      | **92.45%** | **3.11** |

> Note: These are **token-level** accuracy metrics (per character). Word-level accuracy is lower (see Evaluation Results below).

### Step 4: Evaluate & Export

If the Colab session disconnected after training, use `notebooks/evaluate_and_export.ipynb` to:

1. Load checkpoints from Google Drive
2. Run test set evaluation
3. Convert to CTranslate2
4. Download models locally

---

## 📊 Evaluation Results

Evaluated on 15,000 test pairs (5,000 per language) using the step 30,000 checkpoint.

### Word-Level Accuracy & Character Error Rate

| Language          |    Test Pairs    |  Word Accuracy  |       CER       |
| ----------------- | :--------------: | :--------------: | :--------------: |
| Hindi             |      5,000      |      49.16%      |      13.13%      |
| Bengali           |      5,000      |      43.46%      |      15.68%      |
| Tamil             |      5,000      | **58.10%** | **9.25%** |
| **Overall** | **15,000** | **50.24%** | **12.44%** |

- **Word Accuracy**: Percentage of words transliterated **exactly** correctly (strict match)
- **CER**: Character Error Rate — average edit distance normalized by reference length

### Translation Quality

| Metric                         | Value                           |
| ------------------------------ | ------------------------------- |
| Prediction Perplexity          | **1.10** (very confident) |
| Inference Time (15K sentences) | ~54 seconds                     |

> **Note on metrics**: Token-level validation accuracy (92.45%) is much higher than word-level accuracy (50.24%) because a single incorrect character in a word makes the entire word "wrong" at the word level, even if 90% of the characters are correct.

---

## 📝 Sample Outputs

### CTranslate2 Inference Results

| Input     | Hindi           | Bengali           | Tamil           |
| --------- | --------------- | ----------------- | --------------- |
| namaste   | नमस्ते ✅ | —                | —              |
| bharat    | भरत          | —                | —              |
| delhi     | देलही      | —                | —              |
| kolkata   | —              | কোলকাতা ✅ | —              |
| dhanyabad | —              | ধন্যবাদ ✅ | —              |
| chennai   | —              | —                | சென்னை ✅ |
| vanakkam  | —              | —                | வனக்கம்  |

### Test Set Predictions (selected)

| Source                        | Expected                       | Predicted                      | Match |
| ----------------------------- | ------------------------------ | ------------------------------ | :---: |
| `[tam] velippuram`          | வெளிப்புறம்         | வெளிப்புறம்         |  ✅  |
| `[tam] thairiyam`           | தைரியம்                 | தைரியம்                 |  ✅  |
| `[ben] dokaandaarer`        | দোকানদারের           | দোকানদারের           |  ✅  |
| `[hin] westwork`            | वेस्टवर्क             | वेस्टवर्क             |  ✅  |
| `[hin] ubhrega`             | उभरेगा                   | उभरेगा                   |  ✅  |
| `[hin] alba`                | अल्बा                     | अल्बा                     |  ✅  |
| `[tam] kalanthuraiyaadalil` | கலந்துரையாடலில் | கலந்துரையாடலில் |  ✅  |
| `[tam] indiana`             | இந்தியானா             | இந்தியானா             |  ✅  |
| `[ben] procharok`           | প্রচারক                 | প্রচারক                 |  ✅  |
| `[hin] julahon`             | जुलाहों                 | जुलाहों                 |  ✅  |

---

## ⚡ CTranslate2 Optimization

The trained OpenNMT-py model is converted to CTranslate2 with int8 quantization for deployment.

### Model Size Comparison

| Model                    | Size              | Format          |
| ------------------------ | ----------------- | --------------- |
| OpenNMT-py checkpoint    | ~100 MB           | PyTorch float32 |
| CTranslate2 (int8)       | **12.7 MB** | Quantized int8  |
| **Size Reduction** | **~87%**    |                 |

### Convert Model (done in Colab)

```bash
# Fix attention type for CT2 compatibility
python -c "
import torch
cp = torch.load('model.pt', map_location='cpu')
cp['opt'].self_attn_type = 'scaled-dot'
torch.save(cp, 'patched_model.pt')
"

# Convert to CTranslate2
ct2-opennmt-py-converter \
    --model_path patched_model.pt \
    --output_dir models/ct2_model \
    --quantization int8
```

> **Note**: OpenNMT-py 3.5.1 uses `scaled-dot-flash` attention by default, which CTranslate2 doesn't support. The patch above changes the config flag — model weights are identical.

---

## 🌐 Deployment

### Run Locally

```bash
python deploy/app.py --model_dir models/ct2_model
# Opens at http://localhost:7860
```

Options:

- `--model_dir`: Path to CTranslate2 model (default: `models/ct2_model`)
- `--port`: Server port (default: 7860)
- `--share`: Create a public Gradio link

### Deploy to Hugging Face Spaces

The live demo is deployed at: [huggingface.co/spaces/avi705/devnagri-transliteration](https://huggingface.co/spaces/avi705/devnagri-transliteration)

To deploy your own:

1. Create a new Gradio Space on [huggingface.co/new-space](https://huggingface.co/new-space)
2. Clone the Space repository:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   cd YOUR_SPACE
   ```
3. Copy the required files:
   ```bash
   cp deploy/app_hf.py YOUR_SPACE/app.py
   cp deploy/requirements.txt YOUR_SPACE/
   cp -r models/ct2_model YOUR_SPACE/
   ```
4. Track the model binary with Git LFS, commit, and push:
   ```bash
   cd YOUR_SPACE
   git lfs install
   git lfs track "ct2_model/model.bin"
   git add -A
   git commit -m "Deploy transliteration app"
   git push
   ```
5. The app auto-deploys on the free CPU tier

---

## 🏗️ Architecture & Design

### Why Character-Level Transformer?

- Transliteration maps **individual characters** between scripts
- Small vocabulary (~208 tokens) covers all characters across 3 scripts + English
- Compact model (7.5M params) suitable for CPU deployment

### Why Multilingual with Prefix Tokens?

- **One model** instead of 3 separate models
- Cross-lingual transfer — shared phonetic patterns improve all languages
- Simpler deployment and smaller footprint

### Why CTranslate2?

- Native OpenNMT-py converter — zero friction
- int8 quantization: **87% size reduction** with minimal accuracy loss
- Optimized C++ inference: layer fusion, batch reordering
- Runs on CPU without PyTorch dependency

### Noam Learning Rate Schedule

Uses the "Attention Is All You Need" schedule:

```
lr = base_lr × min(step^(-0.5), step × warmup^(-1.5)) × hidden_size^(-0.5)
```

- Warms up linearly for 4000 steps, then decays as `1/√step`
- Peak effective LR ≈ 0.002 at step 4000

See [docs/report.md](docs/report.md) for detailed design decisions, challenges faced, and potential improvements.

---

## 🔧 Known Issues & Fixes

Issues encountered during development and their solutions:

| Issue                                                   | Solution                                               |
| ------------------------------------------------------- | ------------------------------------------------------ |
| `model_type: transformer` invalid in OpenNMT-py 3.5.1 | Use `model_type: text`                               |
| NumPy 2.x incompatible with torch 2.2.x                 | Install `numpy<2`                                    |
| `build_vocab` requires `-save_data` flag            | Add `-save_data models/data`                         |
| `scaled-dot-flash` not supported by CTranslate2       | Patch checkpoint:`opt.self_attn_type = 'scaled-dot'` |
| Colab session disconnect loses files                    | Save checkpoints to Google Drive                       |
| `%%writefile` fails if directory doesn't exist        | Run `mkdir -p` before `%%writefile`                |

---

## 📄 License

This project uses the [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) dataset (CC-BY / CC0 license).

---

## 🙏 Acknowledgments

- [AI4Bharat](https://ai4bharat.org/) for the Aksharantar dataset
- [OpenNMT](https://opennmt.net/) for the training framework
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) for inference optimization
- [Gradio](https://gradio.app/) for the demo framework
- Google Colab for free GPU access
