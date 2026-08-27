# Setup Guide — `.venv` for `yahoo_answers_topic_classification_full.ipynb`

Step-by-step instructions for the local Python environment of the
**Yahoo! Answers 10-class topic classification (COMPLETE RUN)** notebook.

> This notebook uses **PyTorch** (not TensorFlow) and is built for an
> **Intel Arc B580 (XPU)** GPU on Windows. Device auto-detection order:
> CUDA → XPU → MPS → CPU.

---

## 0. What you end up with

| Component | Value |
|---|---|
| Python | 3.12 (pinned in `.python-version`) |
| Virtualenv | `.venv\` at the project root (`D:\MACHINE LEARNING PROJECTS\CSE440`) |
| PyTorch | XPU build (Intel Arc GPU support) |
| Key packages | transformers, datasets, gensim, scikit-learn, nltk, wordcloud, tqdm, pandas, numpy, matplotlib, seaborn |
| Disk usage | ~3 GB env + ~2 GB first-run caches (BERT, GloVe, dataset) |

## 1. Prerequisites

1. **uv** installed (`pip install uv` or the installer from [docs.astral.sh/uv](https://docs.astral.sh/uv/)). Check: `uv --version`.
2. **Intel GPU driver**, recent version — the XPU torch wheel needs an up-to-date Arc driver from Intel.
3. **VS Code** with the **Python** + **Jupyter** extensions installed.

## 2. Create the virtual environment

Open PowerShell in the project root and run:

```powershell
cd "D:\MACHINE LEARNING PROJECTS\CSE440"

# Pin Python 3.12 for this repo (writes .python-version)
uv python pin 3.12

# Create .venv (uv downloads CPython 3.12 automatically if missing)
uv venv
```

## 3. Install PyTorch with Intel Arc (XPU) support

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/xpu
```

> ⚠️ If `.venv` already contains a **CPU** torch (like `2.13.0+cpu`), this step
> replaces it — the regular `torch` wheel from PyPI has **no** Arc GPU support.

**NVIDIA GPU (CUDA) instead?** Just install the default wheel — on Windows it
already includes the CUDA runtime, no extra index needed:

```powershell
uv pip install torch
```

Verify with `torch.cuda.is_available()` (needs a reasonably recent NVIDIA
driver). Only if the driver is old and can't be updated, pin an older CUDA
build explicitly, e.g. `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`.

**No GPU at all?** The plain wheel is fine; the notebook falls back to CPU with
a warning (training will be slow).

> The notebook itself needs **no changes** for any of these — its device
> auto-detection picks CUDA → XPU → MPS → CPU automatically.

## 4. Install the remaining packages

Everything the notebook imports beyond torch:

```powershell
uv pip install jupyter ipykernel transformers datasets gensim wordcloud nltk tqdm scikit-learn pandas numpy matplotlib seaborn
```

(No TensorFlow needed — that's the *other* notebook, `yahoo_answers_topic_classification.ipynb`.)

## 5. Verify the environment

```powershell
uv run python -c "import torch; print(torch.__version__, '| XPU available:', torch.xpu.is_available())"
uv run python -c "import transformers, datasets, gensim, sklearn, nltk, wordcloud; print('imports OK')"
```

Expected: `XPU available: True` and `imports OK`.
(If XPU shows `False`: update the Intel driver and retry — the notebook will
otherwise silently fall back to CPU.)

## 6. Select the kernel in VS Code / Jupyter

1. Open `notebooks\yahoo_answers_topic_classification_full.ipynb`.
2. Click the **kernel picker** (top right of the notebook).
3. Choose **Python Environments… → `.venv (Python 3.12)`**.

The notebook's own package-check cell (it pip-installs anything missing into
the running kernel) then becomes a no-op — everything is already installed.

## 7. First-run downloads (automatic, cached)

| What | Size (approx.) | Cache location |
|---|---|---|
| `bert-base-uncased` weights + tokenizer | ~440 MB | `~\.cache\huggingface\` |
| GloVe `glove-wiki-gigaword-100` via gensim | ~130 MB | `~\gensim-data\` |
| NLTK resources (stopwords etc.) | small | `~\nltk_data\` |
| Yahoo Answers dataset (HF mirror, if no local CSVs) | ~1.5 GB | `~\.cache\huggingface\datasets\` |

## 8. Optional — use the original CSVs instead of the HF download

The notebook auto-searches these locations (first hit wins) before falling
back to the Hugging Face mirror `yassiracharki/Yahoo_Answers_10_categories_for_NLP`:

- `..\Dataset\` (i.e. `CSE440\Dataset\`)
- `Dataset\`
- `data\raw\`

Place the headerless original CSVs (`train.csv`, `validation.csv`, `test.csv`)
in one of those folders to skip the ~1.5 GB download. Expected positional
columns: `0 class_index, 1 question_title, 2 question_content, 3 best_answer`.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `running scripts is disabled` when activating | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry |
| `XPU available: False` | Update Intel Arc driver; confirm step 3 used the `whl/xpu` index |
| `torch.xpu` attribute error | Old/CPU torch in the venv — redo step 3 |
| Training slow + "No GPU/XPU accelerator detected" warning | Torch is running on CPU; check `torch.xpu.is_available()` |
| Out of RAM | The header cell documents reduced sample sizes (`N_PER_CLASS_TRAIN = 2000` etc. — commented in the config cell) |

## 10. Cloud alternative

Per the notebook header: on **Kaggle/Colab** with a CUDA accelerator it runs
unchanged (torch is preinstalled, auto-detect picks `cuda`). On Kaggle attach
the original CSVs as a private dataset or let it fetch the HF mirror.

---

### Quick copy-paste (all of the above, condensed)

```powershell
cd "D:\MACHINE LEARNING PROJECTS\CSE440"
uv python pin 3.12
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/xpu   # Intel Arc (this PC)
# uv pip install torch                                                  # NVIDIA CUDA (default wheel has CUDA built in)
uv pip install jupyter ipykernel transformers datasets gensim wordcloud nltk tqdm scikit-learn pandas numpy matplotlib seaborn
uv run python -c "import torch; print(torch.__version__, '| XPU available:', torch.xpu.is_available())"
```

> No uv on the teammate's machine? Plain venv works too:
> `py -3.12 -m venv .venv` → `.\.venv\Scripts\Activate.ps1` →
> `python -m pip install <same packages>`. And they should `cd` into their own
> clone of the repo — the path above is just this machine's location.
