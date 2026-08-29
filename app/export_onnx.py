"""
Export the fine-tuned BERT classifier to ONNX + int8 quantized ONNX for
in-browser inference (transformers.js) on the static web deployment.

Writes:
  web/model/model.onnx            (fp32, ~440 MB - reference only)
  web/model/model_quantized.onnx  (int8, ~110 MB - what the site loads)
  web/model/{config.json, tokenizer.json, tokenizer_config.json}

Run:  python app/export_onnx.py
"""

import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer, BertForSequenceClassification

SRC = Path(__file__).resolve().parents[1] / 'app' / 'model'
OUT = Path(__file__).resolve().parents[1] / 'web' / 'model'
MAX_LEN = 96

OUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(SRC)
model = BertForSequenceClassification.from_pretrained(SRC)
model.eval()

enc = tokenizer(['a test question', 'another one'], truncation=True,
                padding='max_length', max_length=MAX_LEN, return_tensors='pt')
dummy = (enc['input_ids'], enc['attention_mask'], enc['token_type_ids'])

fp32_path = OUT / 'model.onnx'
print('Exporting fp32 ONNX...')
torch.onnx.export(
    model, dummy, str(fp32_path),
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['logits'],
    dynamic_axes={name: {0: 'batch'} for name in
                  ['input_ids', 'attention_mask', 'token_type_ids', 'logits']},
    opset_version=17,
    dynamo=False,  # legacy exporter: cleaner graph, quantization-friendly
)

print('Quantizing to int8...')
quantize_dynamic(str(fp32_path), str(OUT / 'model_quantized.onnx'),
                 weight_type=QuantType.QInt8)

for fname in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
    shutil.copy(SRC / fname, OUT / fname)

# --- numeric sanity: fp32 torch vs int8 ONNX on real-ish inputs -------------
texts = [
    'how do i fix the blue screen error on windows my pc keeps restarting',
    'i have had a sore throat and mild fever for 7 days should i see a doctor',
    'do you think the current administration policies will help them win reelection',
    'best way to learn calculus for my first year course struggling with derivatives',
]
enc = tokenizer(texts, truncation=True, padding='max_length',
                max_length=MAX_LEN, return_tensors='pt')
with torch.no_grad():
    ref = model(**enc).logits.softmax(-1).numpy()

sess = ort.InferenceSession(str(OUT / 'model_quantized.onnx'),
                            providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {k: v.numpy() for k, v in enc.items()})[0]
ort_prob = torch.tensor(ort_out).softmax(-1).numpy()

top_ref, top_ort = ref.argmax(1), ort_prob.argmax(1)
for i, t in enumerate(texts):
    print(f'  torch={top_ref[i]} int8={top_ort[i]} max_prob_diff='
          f'{np.abs(ref[i] - ort_prob[i]).max():.4f} | {t[:50]}')
print('agreement:', float((top_ref == top_ort).mean()))

fp32_path.unlink()  # keep only the quantized file + configs
print('Done ->', OUT.resolve())
