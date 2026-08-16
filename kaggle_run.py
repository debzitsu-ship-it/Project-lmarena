"""
One-command Kaggle pipeline. From a Kaggle notebook cell, run:

    !git clone -q -b arena/01a0011c-project-lmarena https://github.com/debzitsu-ship-it/Project-lmarena.git
    !cd Project-lmarena && python kaggle_run.py

Then "Save & Run All (Commit)" and close the app — it runs in the background.

Does everything: data download -> tokenizer -> 20M pretrain (time-boxed,
auto-resume) -> chat fine-tune -> export to /kaggle/working/model_output.
From scratch; no pretrained weights; no AI APIs.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

T0 = time.time()
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "10.5"))
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

OUT = "/kaggle/working/model_output" if os.path.isdir("/kaggle/working") else os.path.join(ROOT, "model_output")
os.makedirs(OUT, exist_ok=True)


def sh(mod_args: list[str], check: bool = True) -> int:
    print("+", " ".join(mod_args), flush=True)
    rc = subprocess.run([sys.executable, "-m", *mod_args]).returncode
    if check and rc != 0:
        raise SystemExit(f"step failed: {mod_args} rc={rc}")
    return rc


def hours_left() -> float:
    return TIME_BUDGET_H - (time.time() - T0) / 3600


# ---------------------------------------------------------------- 1. GPU check
try:
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
          else "NONE — enable GPU in Session options for a fast run!")
except Exception as e:  # torch always preinstalled on Kaggle
    raise SystemExit(f"PyTorch missing: {e}")

# ------------------------------------------- 2. restore previous run (resume)
os.makedirs("my_ai/checkpoints", exist_ok=True)
os.makedirs("my_ai/data/processed", exist_ok=True)
restored = []
if os.path.isdir("/kaggle/input"):
    for pat, dest in [("latest.pt", "my_ai/checkpoints/latest.pt"),
                      ("tokenizer.json", "my_ai/data/processed/tokenizer.json")]:
        hits = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
        if hits:
            shutil.copy(hits[0], dest)
            restored.append(hits[0])
print("restored:", restored or "nothing (fresh start)")

# ---------------------------------------------------------------- 3. get data
sh(["my_ai.data.fetch_gitenberg"], check=False)
sh(["my_ai.data.fetch_python_code"], check=False)
for bid in [1080, 2542, 5200, 16389, 902, 408, 1232, 844]:
    dest = f"my_ai/data/raw/gutenberg_{bid}.txt"
    if os.path.exists(dest):
        continue
    for url in (f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
                f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt"):
        try:
            txt = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")
            s, e = txt.find("*** START"), txt.find("*** END")
            if s != -1:
                txt = txt[txt.find("\n", s):]
            if e != -1:
                txt = txt[:e]
            with open(dest, "w") as f:
                f.write(txt)
            break
        except Exception:
            continue
sh(["my_ai.data.make_chat_data"])
total = sum(os.path.getsize(os.path.join("my_ai/data/raw", f)) for f in os.listdir("my_ai/data/raw"))
print(f"corpus: {total / 1e6:.1f} MB")

# ------------------------------------------------- 4. tokenizer + token packs
if os.path.exists("my_ai/data/processed/tokenizer.json") and restored:
    print("resume mode: reusing previous tokenizer, repacking tokens")
    from my_ai.data.dataset import load_corpus, split_corpus, tokenize_and_pack
    from my_ai.tokenizer.tokenizer import load_tokenizer
    tok = load_tokenizer("my_ai/data/processed/tokenizer.json")
    docs = load_corpus(["my_ai/data/raw"])
    tr, va, te = split_corpus(docs)
    meta = {"vocab_size": tok.vocab_size}
    for name, subset in (("train", tr), ("val", va), ("test", te)):
        arr = tokenize_and_pack(subset, tok, f"my_ai/data/processed/{name}.bin")
        meta[f"{name}_tokens"] = int(len(arr))
    with open("my_ai/data/processed/meta.json", "w") as f:
        json.dump(meta, f)
    print(meta)
else:
    sh(["my_ai.prepare_data", "--input", "my_ai/data/raw",
        "--out", "my_ai/data/processed",
        "--vocab-size", "8192", "--tokenizer-sample-chars", "2000000"])
sh(["my_ai.prepare_finetune", "--data", "my_ai/data/processed",
    "--raw", "my_ai/data/raw", "--chat-frac", "0.7"])

# ----------------------------------------------- 5. Stage A: pretrain (boxed)
TOTAL_STEPS, CHUNK = 20000, 2000
done = 0
while done < TOTAL_STEPS and hours_left() > 1.0:
    args = ["my_ai.train", "--config", "my_ai/configs/small_20m.json",
            "--data", "my_ai/data/processed", "--steps", str(CHUNK),
            "--batch-size", "32", "--lr", "6e-4",
            "--sample-prompt", "def add(a, b):"]
    if os.path.exists("my_ai/checkpoints/latest.pt"):
        args += ["--resume", "my_ai/checkpoints/latest.pt"]
    if sh(args, check=False) != 0:
        break
    done += CHUNK
    # keep a rolling safety copy in the output dir
    shutil.copy("my_ai/checkpoints/latest.pt", os.path.join(OUT, "latest.pt"))
    print(f"=== {done} steps this session, {hours_left():.1f}h left ===", flush=True)
print("Stage A complete for this session")

# --------------------------------------------------- 6. Stage B: chat SFT
if os.path.exists("my_ai/checkpoints/latest.pt"):
    sh(["my_ai.finetune", "--base", "my_ai/checkpoints/latest.pt",
        "--data", "my_ai/data/processed", "--steps", "2000",
        "--lr", "1e-4", "--batch-size", "32"], check=False)

# ---------------------------------------------------------------- 7. export
src = "my_ai/checkpoints/chat/best.pt" if os.path.exists("my_ai/checkpoints/chat/best.pt") \
    else "my_ai/checkpoints/latest.pt"
if os.path.exists(src):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({"model_state": ckpt["model_state"], "model_config": ckpt["model_config"],
                "step": ckpt.get("step"), "note": "Kaggle 20M two-stage run, from scratch"},
               os.path.join(OUT, "model_release.pt"))
    sh(["my_ai.inference.export_numpy", "--checkpoint", os.path.join(OUT, "model_release.pt"),
        "--out", os.path.join(OUT, "model_numpy.npz")], check=False)
    tok_src = ("my_ai/checkpoints/chat/tokenizer.json"
               if os.path.exists("my_ai/checkpoints/chat/tokenizer.json")
               else "my_ai/data/processed/tokenizer.json")
    shutil.copy(tok_src, os.path.join(OUT, "tokenizer.json"))
    shutil.copy("my_ai/checkpoints/latest.pt", os.path.join(OUT, "latest.pt"))

    # quality snapshot into the log
    from my_ai.inference.generate import generate_text
    from my_ai.tokenizer.tokenizer import load_tokenizer
    from my_ai.training.trainer import load_checkpoint
    model, _ = load_checkpoint(os.path.join(OUT, "model_release.pt"))
    tok = load_tokenizer(os.path.join(OUT, "tokenizer.json"))
    model.eval()
    for p in ["<|user|>Tell me about yourself.<eos><|assistant|>",
              "<|user|>Write a Python function that adds two numbers.<eos><|assistant|>",
              "<|user|>What is quantum physics?<eos><|assistant|>",
              "Once upon a time"]:
        torch.manual_seed(0)
        out = generate_text(model, tok, p, max_new_tokens=60, temperature=0.7, top_k=40)
        print("PROMPT:", p)
        print("OUTPUT:", out[len(p):].strip()[:300])
        print("---")
    print("DONE. Files in", OUT, ":", os.listdir(OUT))
else:
    print("no checkpoint produced — check the log above for errors")
