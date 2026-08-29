---
title: Yahoo Answers Topic Classification - Base BERT
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Gradio interface - Yahoo! Answers topic classification (Base BERT)

A small web UI where anyone can paste a Yahoo! Answers-style question and get
the predicted topic (10 classes) from the project's best model — the
fine-tuned `bert-base-uncased` from the notebook
(**70.02% test accuracy / 0.696 macro-F1**, best of 11 models).

This folder is intentionally **self-contained and separate from the notebook**:
the notebook is untouched, and everything the interface needs lives here.

```
app/
├── app.py             # the interface (Gradio)
├── export_cell.txt    # the ONE cell to paste at the end of the notebook
│                      #   to export the trained model into app/model/
├── retrain.py         # standalone re-train of the notebook's winning BERT
│                      #   config - use it when no notebook kernel is alive
├── requirements.txt   # interface-only deps (installed into the project .venv)
├── .gitignore         # keeps the 440 MB model dir out of git
└── model/             # created by the export cell or retrain.py (not committed)
```

## 1. Get the model weights (pick one)

The notebook keeps the trained weights in kernel memory only — they are lost
when the kernel shuts down, so `app/model/` must be (re)created once:

**Option A — the notebook kernel is still alive (right after a run):**

1. Open `notebooks/yahoo_answers_topic_classification_full.ipynb`.
2. Add a new cell at the very end, paste the contents of `export_cell.txt`,
   run it. This writes model + tokenizer + label names into `app/model/`.

**Option B — no kernel alive (standalone re-train, no notebook needed):**

```powershell
.\.venv\Scripts\Activate.ps1
python app\retrain.py
```

Reproduces the committed run's winning BERT config exactly (lr=2e-5, batch=32,
epochs=2, seed 42, 40k/5k/10k stratified subsample, lemmatized text, bf16
autocast) and exports straight into `app/model/`, then reports test metrics —
expect ~70.0% accuracy / ~0.696 macro-F1 (the notebook notes runs reproduce
within ~±0.2 pt). ~20–40 min on an RTX 4060; dataset + base BERT come from the
local HF cache, no large downloads.

## 2. Run locally

```powershell
uv pip install -r app/requirements.txt        # into the project .venv
.\.venv\Scripts\Activate.ps1
cd app
python app.py
```

Opens `http://127.0.0.1:7860`. Inference auto-uses CUDA when available
(RTX 4060 here), falling back to CPU.

## 3. Deploy to Hugging Face Spaces (free, recommended)

The bonus criteria accept "Vercel **or any other platform**". Vercel is a bad
fit: it's serverless with tight size limits and `bert-base-uncased` is
~440 MB. HF Spaces hosts Gradio apps natively:

1. Create a **Space** at huggingface.co/new-space → SDK: **Gradio**, CPU (free).
2. Upload `app.py` and `requirements.txt` from this folder.
3. Ship the model — pick one:
   - **via the Hub (recommended):** uncomment the `push_to_hub` lines in
     `export_cell.txt`, run them in the notebook, then in the Space
     **Settings → Variables and secrets** add a *secret*
     `MODEL_ID = YOUR_USERNAME/yahoo-answers-bert`. `app.py` reads it
     automatically. Nothing big is ever committed.
   - **or commit the files:** upload `app/model/*` to the Space repo (the web
     UI handles LFS for the ~440 MB weights automatically).
4. The Space builds in ~2–3 min → that public URL is the submission link for
   the Google Form (§7, "link of your project deployed ... for bonus").

## Notes

- The prediction path replicates the notebook exactly: `build_text`
  (title + content) → `preprocess_lemmatized` (the strategy the notebook
  selected on validation) → tokenizer `max_len=96` → softmax over 10 classes.
  The "Model input (preprocessed)" box in the UI exists precisely so the viva
  can show this parity.
- If you export **BERT v2** instead of v1 (see the commented block in
  `export_cell.txt`), `app.py` must be adapted — v2 was trained on raw
  pair-encoded `[CLS] title [SEP] content [SEP]` inputs, not lemmatized text.
