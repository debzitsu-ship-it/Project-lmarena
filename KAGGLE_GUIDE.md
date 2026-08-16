# Kaggle background training — voice-friendly guide

Kaggle runs the whole training on ITS servers after one "Save & Run All".
You can close the app / switch apps / power off the phone — it keeps going
(up to ~12 h per run, ~30 GPU-hours free per week).

## One-time setup (the only fiddly part)

1. Create a free account at kaggle.com (email + password).
2. Verify your phone number (Settings → Phone verification) — required for GPU.
   Voice tip: both forms work with voice keyboards; if it's painful, this
   5-minute step is a good thing to ask any visiting person to do.

## Each training run (voice-doable, ~6 actions)

1. Open kaggle.com → Create → Notebook.
2. In the notebook: File → Import Notebook → GitHub tab → paste:
   `debzitsu-ship-it/Project-lmarena` → pick `kaggle_train.ipynb`.
3. Right panel → Session options:
   - Accelerator: **GPU T4 x2** (or any GPU offered)
   - Internet: **On**
4. Press **Save Version** → choose **Save & Run All (Commit)** → Save.
5. DONE. Close the app. Training runs in the background on Kaggle.
6. Come back later: notebook page → **Output** tab → files:
   - `model_release.pt`  (the trained model)
   - `model_numpy.npz` + `tokenizer.json` (phone/Termux version)
   - `latest.pt` (raw checkpoint for resuming)

## Resuming / training MORE

Next session: open your notebook → **Add Input** → your own notebook's
previous output → Save & Run All again. Cell 1 finds `latest.pt` from the
input automatically and continues training from there. Repeat forever —
each run adds ~10 GPU-hours of training.

## What the run produces

A ~20M-parameter from-scratch Transformer (books + Python code, two-stage:
pretrain + chat fine-tune), exported both for PyTorch and as a pure-NumPy
file that runs in Termux on a phone with no PyTorch.

## Honest expectations

20M params + ~50 MB corpus = fluent-ish English, decent chat behavior,
code-shaped Python, honest "I don't know". Clearly better than the CPU
models in this repo. Still far from ChatGPT — that's a matter of thousands
of GPUs, not of this recipe.
