# Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding

> by [Lin Long](https://llong-cs.github.io/llongme/), [Changdae Oh](https://changdaeoh.github.io/), [Seongheon Park](https://seongheon-96.github.io/), and [Sharon Li](https://pages.cs.wisc.edu/~sharonli/).

## Overview

This repository provides tools and scripts to analyze the language prior in Vision-Language Models by examining representation distances across different layers.

## Quick Start

### 1. Data Preparation

First, set up the environment variable for your data path:

```bash
export DATA_PATH=/path/to/your/data
```

Create dataset files in JSONL format under the `DATA_PATH` directory. Each dataset should be named as `{dataset}.jsonl`, where each line contains:

- `image`: the image path or base64 string starting with "data:image/"
- `instruction`: the instruction text
- `target_tokens`: the target tokens (e.g., ["Yes", "No"] or ["A", "B", "C", "D"])
- `other keys`: additional keys you want to include

**Example JSONL entry:**
```json
{"image": "/path/to/image.jpg", "instruction": "What color is the sky?", "target_tokens": ["A", "B", "C", "D"], "answer": "A"}
```

We provide reference data processing scripts for several datasets in the `data_preparation/` folder.

### 2. Generate Hidden States

Generate hidden states for your model. Using Qwen2.5-VL as an example:

```bash
CUDA_VISIBLE_DEVICES=0 python generation/gen_qwenvl.py --dataset mme
```

**Multi-GPU Support:**
This step supports multi-GPU parallel generation. After generation is complete, you need to merge the results:

```bash
python utils/merge.py
```

### 3. Plot Representation Distance

Use the plotting script to visualize representation distance curves:

```bash
python plot_divergences.py --model qwenvl --dataset mme
```

**Available options:**
- `--model`: Model name (e.g., qwenvl, llava, gemma)
- `--dataset`: Dataset name (e.g., mme, mmbench, vlind)
- `--data_path`: Path to the data directory (default: "data")

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```
