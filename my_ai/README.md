# my_ai — a language model trained truly from scratch

A small decoder-only Transformer whose weights start as random numbers and are
trained by us, on our data, on our machine. **No OpenAI/Claude/Gemini APIs, no
pretrained weights, no external AI services anywhere in this codebase** — the
only ML dependency is PyTorch as a tensor/autograd library. Grep the code:
there are no network calls at inference or chat time at all.

## Layout

```
my_ai/
├── data/            dataset pipeline (.txt/.json/.jsonl -> clean -> split -> pack -> stream)
│   ├── dataset.py
│   ├── make_chat_data.py   synthetic chat-format data from hand-written templates
│   └── raw/                put your training text here
├── tokenizer/       CharTokenizer + byte-level BPE, both from scratch, trainable
├── model/           the Transformer (embeddings, causal multi-head attention, FFN, ...)
├── training/        AdamW + warmup/cosine LR + clipping + AMP + checkpoints
├── inference/       greedy / temperature / top-k / top-p / repetition penalty / stop tokens
├── chat/            local CLI chat + JSON-file memory (facts, history, pins)
├── checkpoints/     best.pt / latest.pt / tokenizer.json (gitignored)
├── configs/         nano_cpu (3M) ... xl_1b (1.2B) model configs
├── tests/           32 pytest tests covering every stage
├── hardware.py      inspects CPU/RAM/GPU/VRAM/disk, recommends a config
├── prepare_data.py  stage 5 entry point
├── train.py         stage 6 entry point
├── evaluate.py      loss / perplexity / tokens-sec / params on held-out data
└── requirements.txt
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r my_ai/requirements.txt

# 0. what can this machine handle?
.venv/bin/python -m my_ai.hardware

# 1. run all tests
.venv/bin/pytest my_ai/tests -q

# 2. put .txt/.json/.jsonl files in my_ai/data/raw/ (a starter corpus is included),
#    optionally regenerate the synthetic chat data:
.venv/bin/python -m my_ai.data.make_chat_data

# 3. train tokenizer + pack tokens
.venv/bin/python -m my_ai.prepare_data --input my_ai/data/raw --vocab-size 1024

# 4. train (pick a config, or --auto)
.venv/bin/python -m my_ai.train --config my_ai/configs/nano_cpu.json --steps 2500 --lr 1e-3

# 5. evaluate on the held-out test split
.venv/bin/python -m my_ai.evaluate --split test

# 6. chat, fully locally
.venv/bin/python -m my_ai.chat.cli
```

## How it works, honestly

- **Parameters** are just numbers in matrices. "Training" nudges each one,
  millions of times, in the direction that makes the training text less
  surprising (lower cross-entropy). Nothing else happens.
- **Embeddings**: each token ID indexes a learned vector. Tokens used in
  similar contexts end up with similar vectors because that reduces loss.
- **Attention**: every position computes a weighted average over earlier
  positions' values, with weights from query·key similarity. The causal mask
  forbids looking at the future, which keeps next-token prediction honest.
- **Why small models write poor text**: with ~3M parameters and ~60K training
  tokens, the model can learn spelling, common words, local grammar, and the
  chat format — but it cannot store world knowledge or long-range reasoning.
  It is a statistical parrot of its tiny corpus. That is not a bug; it is
  what this amount of data and compute buys. More data teaches more patterns;
  more parameters give more room to store them; better data quality means
  the patterns learned are patterns worth having.
- **"Memory" in the chat app is not learning.** Facts and history live in
  JSON files and are pasted into the prompt each turn. The weights change
  only during training. `Memory.export_finetune_jsonl()` exists so selected
  conversations can later become fine-tuning data — that is the honest path
  from memory to knowledge.

## Scaling roadmap

| Config       | Params | Ctx  | Hardware needed          | Data wanted        |
|--------------|--------|------|--------------------------|--------------------|
| nano_cpu     | ~3.4M  | 128  | any laptop CPU           | 1–10 MB text       |
| tiny_5m      | ~5M    | 256  | laptop CPU (hours)       | 10–50 MB           |
| small_20m    | ~20M   | 512  | 6 GB GPU / M-series      | 100 MB – 1 GB      |
| base_50m     | ~50M   | 1024 | 10 GB GPU, ~a day        | 1–5 GB (≈1B tok)   |
| medium_100m  | ~110M  | 1024 | 20 GB GPU, days          | ≈2–10 B tokens     |
| large_300m   | ~350M  | 2048 | A100-40GB, ~a week       | ≈10–30 B tokens    |
| xl_1b        | ~1.2B  | 2048 | 8×A100 + FSDP, weeks     | 20 B+ tokens       |

Rule of thumb (Chinchilla): ~20 training tokens per parameter for
compute-optimal pretraining. Below ~1B params, data quality and enough
epochs matter more than exotic tricks.
