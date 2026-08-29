---
license: apache-2.0
base_model: bert-base-uncased
pipeline_tag: text-classification
library_name: transformers.js
tags:
- yahoo-answers
- topic-classification
---

# Yahoo! Answers topic classification (Base BERT, fine-tuned)

10-class classifier for Yahoo! Answers questions (title + content): Society &
Culture, Science & Mathematics, Health, Education & Reference, Computers &
Internet, Sports, Business & Finance, Entertainment, Family & Relationships,
Politics & Government.

Winner of 11 models trained in the CSE440 lab project notebook:
**test 70.02% accuracy / 0.696 macro-F1** (10k stratified official-test subsample).
This copy is int8-quantized ONNX for in-browser inference with
[transformers.js](https://huggingface.co/docs/transformers.js) (dtype `q8`),
reproducing the torch model within 0.1 pt (0.6990 measured).

Training: lr=2e-5, batch=32, 2 epochs, seed 42, lemmatized text, max_len=96.
