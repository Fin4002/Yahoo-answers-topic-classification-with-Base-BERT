# Web deployment — Yahoo! Answers topic classification (Base BERT, in-browser)

Static single-page app that runs the project's fine-tuned BERT classifier
**entirely in the visitor's browser** via
[transformers.js](https://huggingface.co/docs/transformers.js) (ONNX Runtime
Web, int8-quantized weights, ~105 MB, cached after first visit). No server,
no API, no cold starts — it matches the assignment's "hosting platform such
as Vercel" bonus wording with zero running cost.

## Fidelity (measured, not assumed)

| Variant | Test acc | Macro-F1 |
|---|---|---|
| torch fp32, notebook-exact preprocessing (Gradio app) | 0.7003 | 0.6981 |
| ONNX int8, notebook-exact preprocessing | 0.6990 | 0.6973 |
| ONNX int8, JS preprocessing (no lemmatization) — **what runs here** | 0.6993 | 0.6984 |

Measured by `app/measure_js_parity.py` on the notebook's 10k test subsample;
the JS variant skips WordNet lemmatization (not available in browsers), which
costs nothing measurable (94.4% top-1 agreement with the lemmatized variant).

## Files

```
web/
├── index.html        # the page
├── app.js            # model load (dtype q8) + classify + render
├── preprocess.js     # exact port of the notebook's preprocessing (minus lemma)
├── stopwords.js      # NLTK English stopword list (generated)
├── style.css         # dataviz-validated blue ordinal palette, light+dark
├── test.mjs          # Node end-to-end test (same library as the browser)
├── package.json      # only for the Node test / local tooling
└── model/            # exported by app/export_onnx.py (onnx weights gitignored)
```

## Run locally

```powershell
cd web
node test.mjs                 # end-to-end test (Node, no browser)
python -m http.server 8080    # then open http://localhost:8080
```

(ES modules require http, not file://.)

## Deploy (Vercel)

The 105 MB `model/onnx/model_quantized.onnx` exceeds GitHub's 100 MB file
limit, so deploy **from this folder via the Vercel CLI** rather than a GitHub
import (the file is gitignored):

```powershell
npm i -g vercel   # or: npx vercel
cd web
vercel --prod     # first run: log in with GitHub, accept defaults
```

Static project, no build command, no output directory changes.

## Regenerating the model artifacts

```powershell
python app\export_onnx.py        # torch -> ONNX fp32 -> int8 quantized
python app\measure_js_parity.py  # re-verify preprocessing parity
node web\test.mjs                # re-verify the JS pipeline
```
