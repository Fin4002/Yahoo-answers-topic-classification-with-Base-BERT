"""
Measure how much skipping WordNet lemmatization costs when preprocessing is
re-implemented in JavaScript for the in-browser deployment.

Variant A (notebook-exact): minimal -> punctuation strip -> stopwords -> lemma
Variant B (JS-planned):      minimal -> punctuation strip -> stopwords   (no lemma)

Runs the int8 ONNX model on the notebook's exact 10k test subsample for both
variants and reports accuracy + top-1 agreement.

Run:  python app/measure_js_parity.py
"""

import re
import string
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

import nltk

SEED = 42
DATASET_ID = 'yassiracharki/Yahoo_Answers_10_categories_for_NLP'
LABEL_COL = 'class_index'
TEXT_COLS = ['question_title', 'question_content']
N_PER_CLASS_TEST = 1000
MAX_LEN = 96
BATCH = 256

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'web' / 'model'

for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)
LEMMATIZER = WordNetLemmatizer()
STOPWORDS_EN = set(nltk_stopwords.words('english'))


def preprocess_minimal(text):
    t = text.lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'&\w+;', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def preprocess_standard(text):
    t = preprocess_minimal(text).translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in t.split() if w and w not in STOPWORDS_EN]
    return ' '.join(tokens)


def preprocess_A(text):  # notebook-exact (lemmatized)
    return ' '.join(LEMMATIZER.lemmatize(w) for w in preprocess_standard(text).split())


def preprocess_B(text):  # JS-replicable (no lemma)
    return preprocess_standard(text)


def build_text(frame):
    merged = frame[TEXT_COLS[0]].fillna('').astype(str)
    for col in TEXT_COLS[1:]:
        merged = merged + ' ' + frame[col].fillna('').astype(str)
    return merged.str.replace(r'\s+', ' ', regex=True).str.strip()


def stratified_draw(frame, per_class, seed):
    rng = np.random.RandomState(seed)
    picked = []
    for code in sorted(frame[LABEL_COL].unique().tolist()):
        candidates = np.where(frame[LABEL_COL].values == code)[0]
        rng.shuffle(candidates)
        picked.extend(candidates[:per_class].tolist())
    return frame.iloc[picked].reset_index(drop=True)


print('Loading dataset...')
raw = load_dataset(DATASET_ID)
test_full = raw['test'].select_columns([LABEL_COL] + TEXT_COLS).to_pandas()
del raw
test_part = stratified_draw(test_full, N_PER_CLASS_TEST, SEED)
del test_full

le = LabelEncoder().fit(sorted(test_part[LABEL_COL].unique()))
y_test = le.transform(test_part[LABEL_COL])
test_part['text'] = build_text(test_part)

print('Preprocessing A (lemmatized) and B (JS, no lemma)...')
texts_A = test_part['text'].apply(preprocess_A).values
texts_B = test_part['text'].apply(preprocess_B).values

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
sess = ort.InferenceSession(str(MODEL_DIR / 'model_quantized.onnx'),
                            providers=['CPUExecutionProvider'])

def predict_batched(texts):
    preds = []
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(list(texts[i:i + BATCH]), truncation=True,
                        padding='max_length', max_length=MAX_LEN,
                        return_tensors='np')
        logits = sess.run(None, {k: v for k, v in enc.items() if k in
                                 ('input_ids', 'attention_mask', 'token_type_ids')})[0]
        preds.append(logits.argmax(1))
    return np.concatenate(preds)

pred_A = predict_batched(texts_A)
pred_B = predict_batched(texts_B)

print(f'A (lemmatized)  : acc={accuracy_score(y_test, pred_A):.4f} '
      f'macro_f1={f1_score(y_test, pred_A, average="macro"):.4f}')
print(f'B (JS no-lemma) : acc={accuracy_score(y_test, pred_B):.4f} '
      f'macro_f1={f1_score(y_test, pred_B, average="macro"):.4f}')
print(f'top-1 agreement A vs B: {(pred_A == pred_B).mean():.4f}')
