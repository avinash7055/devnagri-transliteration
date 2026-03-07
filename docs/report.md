# Technical Report — Devnagri Multilingual Transliteration

## Overview

Devnagri is a multilingual transliteration system that converts English/Romanized text into Hindi, Bengali, and Tamil scripts using a character-level Transformer model. The model is trained on the Aksharantar dataset using OpenNMT-py and optimized with CTranslate2 for efficient CPU inference.

---

## Design Decisions

### 1. Model Architecture: Character-Level Transformer

We chose a **character-level Transformer** (encoder-decoder) instead of a word/subword-level model because:
- Transliteration operates at the **character level** — mapping individual characters between scripts
- Character-level models capture phonetic patterns more naturally than BPE/subword models
- Small vocabulary (**208 tokens**) covers all Indic script characters + English alphabet + language prefixes
- The model is compact (**7.5M parameters**, 4 layers, 256 hidden dim, 4 heads) — suitable for fast CPU inference

**Model Architecture Details:**
| Component | Details |
|-----------|---------|
| Encoder | 4× TransformerEncoderLayer (256-dim, 4 heads, 1024 FFN) |
| Decoder | 4× TransformerDecoderLayer (256-dim, 4 heads, 1024 FFN) |
| Embeddings | 256-dim, shared between encoder/decoder, positional encoding |
| Vocabulary | 208 tokens (shared src/tgt) |
| Total Parameters | 7,458,000 (Encoder: 3.2M, Decoder: 4.3M) |

### 2. Framework: OpenNMT-py

We selected **OpenNMT-py** over pretrained models (mT5, mBART) because:
- OpenNMT-py is purpose-built for sequence-to-sequence tasks like transliteration
- **Native CTranslate2 support** via `ct2-opennmt-py-converter` — zero friction for optimization
- Pretrained models (mT5/mBART) are 300MB–1GB+, overkill for character-level word transliteration
- Full control over architecture and training hyperparameters
- Active community and well-documented configuration system

### 3. Multilingual Approach: Language Prefix Tokens

Instead of training 3 separate models, we use a **single multilingual model** with language prefix tokens (`<hin>`, `<ben>`, `<tam>`):
- Reduces deployment complexity (1 model instead of 3)
- Enables **cross-lingual transfer** — patterns learned from one language help others
- The model learns shared phonetic representations across scripts
- Source format: `<hin> n a m a s t e` → Target: `न म स ् त े`

### 4. CTranslate2 with int8 Quantization

We chose **int8 quantization** for optimization because:
- **87% size reduction**: 100 MB → 12.7 MB
- Runs efficiently on CPU (important for HF Spaces free tier)
- Minimal accuracy loss from quantization
- No PyTorch dependency required for inference

---

## Training Details

### Dataset: Aksharantar

The [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) dataset from AI4Bharat contains millions of transliteration pairs across 20+ Indic languages.

**Data Split (used in this project):**
| Split | Per Language | Total |
|-------|-------------|-------|
| Train | 100,000 | 300,000 |
| Val | 5,000 | 15,000 |
| Test | 5,000 | 15,000 |

### Training Configuration

- **Platform**: Google Colab (Tesla T4 GPU, 15 GB VRAM)
- **Framework**: OpenNMT-py 3.5.1, PyTorch 2.2.2+cu121
- **Optimizer**: Adam with Noam learning rate schedule
- **Warmup**: 4,000 steps (linear warmup, then sqrt decay)
- **Effective Batch**: 16,384 tokens (4096 × 4 accumulation steps)
- **Total Training Time**: ~3 hours on T4 GPU
- **Total Steps**: 30,000

### Training Curve

| Step | Train Acc | Train PPL | Val Acc | Val PPL | Time |
|------|-----------|-----------|---------|---------|------|
| 100 | 3.4% | 200.1 | — | — | 1 min |
| 1,000 | 69.7% | 6.1 | 74.25% | 5.26 | 6 min |
| 2,000 | 88.2% | 3.3 | 85.84% | 3.64 | 13 min |
| 3,000 | 90.0% | 3.1 | 87.50% | 3.43 | 19 min |
| 5,000 | 93.0% | 2.8 | 89.62% | 3.24 | 32 min |
| 10,000 | ~94% | ~2.7 | ~91% | ~3.15 | ~65 min |
| 30,000 | 92.91% | 2.94 | **92.45%** | **3.11** | ~180 min |

The model converges rapidly in the first 5,000 steps (74% → 90% val accuracy), then gradually improves. The Noam schedule peaks at step 4,000 and decays smoothly.

---

## Evaluation Results

### Word-Level Metrics (Test Set — 15,000 pairs)

| Language | Test Pairs | Word Accuracy | CER (%) |
|----------|:----------:|:-------------:|:-------:|
| Hindi | 5,000 | 49.16% | 13.13% |
| Bengali | 5,000 | 43.46% | 15.68% |
| Tamil | 5,000 | **58.10%** | **9.25%** |
| **Overall** | **15,000** | **50.24%** | **12.44%** |

### Why Token Accuracy ≠ Word Accuracy

- **Token-level** (during training): 92.45% — per-character correctness
- **Word-level** (evaluation): 50.24% — entire word must be exactly right
- A word like "भारत" where one out of 5 characters is wrong = 80% token accuracy but 0% word accuracy
- CER (12.44%) better captures the "almost correct" predictions

### Model Confidence

| Metric | Value |
|--------|-------|
| Prediction Perplexity | 1.10 |
| Inference Time (15K sentences, GPU) | ~54 sec |

---

## Challenges Faced

### 1. NumPy Version Conflict
**Problem**: torch 2.2.x is incompatible with numpy 2.x (causes `_ARRAY_API not found` error).
**Solution**: Install `numpy<2` after OpenNMT-py installation.

### 2. OpenNMT-py 3.5.1 Configuration
**Problem**: `model_type: transformer` is not valid in OpenNMT-py 3.5.1.
**Solution**: Use `model_type: text` with `encoder_type: transformer` and `decoder_type: transformer`.

### 3. `build_vocab` Missing Flag
**Problem**: `onmt.bin.build_vocab` requires `-save_data` argument in 3.5.1.
**Solution**: Add `-save_data models/data` to the command.

### 4. Colab Session Disconnect
**Problem**: First training run completed (30K steps, 92.45% val accuracy), but Colab session disconnected before evaluation. All local files were lost.
**Solution**: Second training run saved checkpoints to **Google Drive** (`save_model: /content/drive/MyDrive/devnagri/models/transliteration_model`). Created `evaluate_and_export.ipynb` for post-training evaluation.

### 5. CTranslate2 Attention Type
**Problem**: OpenNMT-py 3.5.1 defaults to `scaled-dot-flash` attention, which CTranslate2 doesn't support.
**Solution**: Patch the checkpoint's `opt.self_attn_type` from `scaled-dot-flash` to `scaled-dot` before conversion. Model weights are identical — flash attention is a GPU optimization only.

### 6. Dataset Field Name Variability
**Problem**: Aksharantar dataset uses different field names across versions (`native word` vs `native_word`).
**Solution**: Handle multiple possible field names in the preprocessing script.

---

## Potential Improvements

1. **Use more training data**: Current training uses 100K pairs/language from 1M+ available. Using 500K+ pairs would significantly improve word accuracy.
2. **Increase model size**: Adding more layers (6+6) or wider hidden dim (512) could improve quality.
3. **Continue training**: Extend from 30K to 60K+ steps for marginal gains.
4. **Beam search + n-best**: Return top-N transliteration candidates for user selection.
5. **More languages**: Extend to all 20+ Aksharantar languages with same architecture.
6. **Fine-tuning on domain data**: Adapt for specific domains like proper nouns, place names.
7. **Bidirectional transliteration**: Train reverse models (Indic → English).
8. **Attention visualization**: Show which source characters influence each target character.

---

## Tools & Technologies

| Tool | Usage |
|------|-------|
| OpenNMT-py 3.5.1 | Model training |
| PyTorch 2.2.2 | Deep learning framework |
| CTranslate2 4.7.1 | Model optimization & inference |
| Gradio 4.x | Interactive demo UI |
| Google Colab (T4 GPU) | Training environment |
| Aksharantar Dataset | Training data (AI4Bharat) |
