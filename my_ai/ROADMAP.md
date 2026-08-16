# Training roadmap: from parrot to personal AI

## Owner's target spec (recorded 2026-08) and honest feasibility

Wanted: math, general knowledge, coding, agent functions (phone control,
web/image search), huge conversation skills, learns from mistakes,
remembers things, ~2B parameters.

Status of each, honestly:
- **Math**: solved TODAY via the calculator tool (exact at any size) +
  math dataset so the model also learns the language of math.
- **General knowledge**: knowledge dataset now; scales with corpus size.
  Real breadth needs Wikipedia-scale data (see Phase 2).
- **Coding**: code corpus + code Q&A now; useful generation begins ~100M+
  params with 10-100x more code data.
- **Agent functions**: app-level tools, not model weights. Calculator done;
  web search/phone control are integrations to add to the chat app —
  they work with ANY model size.
- **Learns from mistakes**: `learn: question => answer` command stores
  corrections; prepare_finetune folds them into the next fine-tune (x20
  weight). True online learning does not exist for any LLM.
- **Remembers**: memory system (facts/history) exists; corrections persist.
- **2B parameters**: NOT reachable on free compute. 2B needs ~40B+ training
  tokens (~100 GB text) and weeks on 8x A100 (thousands of dollars).
  Kaggle's free ceiling: ~50-125M params trained properly across multiple
  12h sessions. That is the honest maximum on this budget — and a
  well-fed 50-125M model is dramatically better than today's 14M.

Path: 14M (done) -> 50M (next Kaggle runs) -> 125M (chained sessions)
-> beyond requires paid GPUs.

Data is the food. Weights are the memory. This file turns the phased plan
into concrete, runnable steps for THIS repo. Do them in order — each phase
builds on the previous checkpoint.

## Phase 1 — Language (grammar, vocabulary, patterns)

**Feed:** clean books and articles. Public-domain sources are safest:
Project Gutenberg, Wikisource, your own writing.

**Amount:** 100 MB–1 GB of clean text is the sweet spot for a first real
model — big enough to force generalization, small enough to iterate on.
(Our current corpus is ~0.4 MB; the Colab notebook's ~30 books gets you
to ~15 MB; add more IDs to grow it.)

**Run:**
```bash
python -m my_ai.data.ingest_urls --list my_urls.txt        # or drop files in data/raw/
python -m my_ai.prepare_data --input my_ai/data/raw --vocab-size 4096
python -m my_ai.train --config my_ai/configs/small_20m.json --steps 20000
```
**Done when:** samples in the training log read like fluent-ish sentences,
not word salad. Val perplexity dropping steadily.

## Phase 2 — Knowledge (facts)

**Feed:** Wikipedia Simple English dump, documentation, educational text.
Add it to `data/raw/` and CONTINUE training (`--resume`):
```bash
python -m my_ai.train --resume my_ai/checkpoints/latest.pt --steps 20000 ...
```
⚠️ If you retrain the tokenizer (bigger vocab for bigger corpus), you must
restart the model from scratch — a model is married to its tokenizer.

## Phase 3 — Instruction following

**Feed:** many examples of `<|user|>question<eos><|assistant|>answer<eos>`.
`data/make_chat_data.py` generates template-based ones; the real upgrade is
thousands of *diverse, hand-curated or openly licensed* Q/A pairs written
into `.jsonl` files with a `"text"` field using the role markers.
Train on a MIX (~90% plain text, ~10% chat) so it keeps its language skills.

## Phase 4 — Reasoning

**Feed:** worked step-by-step solutions (math word problems, logic,
how-to explanations) in the same chat format, with the reasoning written
out in the assistant turns. At small scale, expect pattern-imitation of
reasoning, not real reasoning — be honest with yourself about the ceiling.

## Phase 5 — Your personal AI

**Feed:** your own selected conversations and documents.
`Memory.export_finetune_jsonl()` dumps chat history into training format:
```bash
python - <<'EOF'
from my_ai.chat.memory import Memory
Memory().export_finetune_jsonl("my_ai/data/raw/my_conversations.jsonl")
EOF
```
Then fine-tune from the latest checkpoint with a LOW learning rate
(`--lr 5e-5`) for a few hundred steps so it adapts without forgetting.

## Rules of thumb

- Quality beats quantity. One clean book beats ten scraped pages of junk.
- ~20 tokens per parameter for compute-optimal pretraining (Chinchilla).
- Never mix tokenizers between phases. One tokenizer, one model lineage.
- Keep `val.bin` honest: if val loss rises while train loss falls,
  you are memorizing, not learning — get more/varied data.
- Licenses matter: public domain (Gutenberg), CC-BY (Wikipedia, cite it),
  your own writing — all fine. Random scraped content — check first.
