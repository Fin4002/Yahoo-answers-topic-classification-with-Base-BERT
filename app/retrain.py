"""
Standalone re-train of the notebook's winning BERT Base configuration.

Reproduces Section 9 of notebooks/yahoo_answers_topic_classification_full.ipynb
- same seed, same stratified subsample, same preprocessing, same training loop -
and exports the fine-tuned model into app/model/ for the Gradio interface.

Winning config (from the committed run's tuning log):
  lr=2e-5, batch_size=32, epochs=2  ->  val 0.7006 acc / 0.6989 macro-F1
                                       test 0.7002 acc / 0.6960 macro-F1
The notebook itself notes runs reproduce within ~ +/-0.2 pt.

Usage (from the repo root, project venv active):
  python app/retrain.py            # trains, evaluates on test, exports
"""

import random
import re
import string
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification

import nltk

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_ID = 'yassiracharki/Yahoo_Answers_10_categories_for_NLP'
LABEL_COL = 'class_index'
TEXT_COLS = ['question_title', 'question_content']

N_PER_CLASS_TRAIN = 4000
N_PER_CLASS_VAL = 500
N_PER_CLASS_TEST = 1000

BERT_NAME = 'bert-base-uncased'
BERT_MAX_LEN = 96
LR = 2e-5
BATCH_SIZE = 32
EPOCHS = 2

OUT_DIR = Path(__file__).resolve().parent / 'model'

TOPIC_NAMES_BY_CODE = {
    1: 'Society & Culture',
    2: 'Science & Mathematics',
    3: 'Health',
    4: 'Education & Reference',
    5: 'Computers & Internet',
    6: 'Sports',
    7: 'Business & Finance',
    8: 'Entertainment',
    9: 'Family & Relationships',
    10: 'Politics & Government',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_DTYPE = torch.bfloat16 if DEVICE.type in ('cuda', 'xpu') else torch.float32

for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)
LEMMATIZER = WordNetLemmatizer()
STOPWORDS_EN = set(nltk_stopwords.words('english'))


# --- preprocessing: exact copies of the notebook's selected strategy --------

def preprocess_minimal(text):
    t = text.lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'&\w+;', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def preprocess_standard(text):
    t = preprocess_minimal(text).translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in t.split() if w and w not in STOPWORDS_EN]
    return ' '.join(tokens)


def preprocess_lemmatized(text):
    tokens = [LEMMATIZER.lemmatize(w) for w in preprocess_standard(text).split()]
    return ' '.join(tokens)


def build_text(frame):
    merged = frame[TEXT_COLS[0]].fillna('').astype(str)
    for col in TEXT_COLS[1:]:
        merged = merged + ' ' + frame[col].fillna('').astype(str)
    return merged.str.replace(r'\s+', ' ', regex=True).str.strip()


# --- data: same seeded stratified subsample as the notebook ------------------

def stratified_draw(frame, per_class, seed):
    rng = np.random.RandomState(seed)
    picked = []
    for code in LABEL_CODES:
        candidates = np.where(frame[LABEL_COL].values == code)[0]
        rng.shuffle(candidates)
        picked.extend(candidates[:per_class].tolist())
    return frame.iloc[picked].reset_index(drop=True)


print(f'Device: {DEVICE}')
print('Loading dataset (HF cache is used if present)...')
raw = load_dataset(DATASET_ID)
train_full = raw['train'].select_columns([LABEL_COL] + TEXT_COLS).to_pandas()
test_full = raw['test'].select_columns([LABEL_COL] + TEXT_COLS).to_pandas()
del raw

LABEL_CODES = sorted(train_full[LABEL_COL].unique().tolist())
print('label codes:', LABEL_CODES, '| train rows:', len(train_full), '| test rows:', len(test_full))

train_val_df = stratified_draw(train_full, N_PER_CLASS_TRAIN + N_PER_CLASS_VAL, SEED)
shuffled = train_val_df.sample(frac=1.0, random_state=SEED)
val_part = shuffled.groupby(LABEL_COL, group_keys=False).head(N_PER_CLASS_VAL)
train_part = shuffled.drop(val_part.index)
test_part = stratified_draw(test_full, N_PER_CLASS_TEST, SEED)
del train_full, test_full, train_val_df, shuffled

print('train:', train_part.shape, '| val:', val_part.shape, '| test:', test_part.shape)

label_encoder = LabelEncoder().fit(train_part[LABEL_COL])
CLASS_NAMES = [TOPIC_NAMES_BY_CODE[code] for code in label_encoder.classes_]
y_train = label_encoder.transform(train_part[LABEL_COL])
y_val = label_encoder.transform(val_part[LABEL_COL])
y_test = label_encoder.transform(test_part[LABEL_COL])
print('classes:', dict(enumerate(CLASS_NAMES)))

for part in (train_part, val_part, test_part):
    part['text'] = build_text(part)
X_train_text = train_part['text'].apply(preprocess_lemmatized).values
X_val_text = val_part['text'].apply(preprocess_lemmatized).values
X_test_text = test_part['text'].apply(preprocess_lemmatized).values
print('preprocessed example:', X_train_text[0][:120])

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t = torch.tensor(y_val, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)


# --- tokenization + training loop: exact copies of the notebook --------------

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)

def bert_tensors(texts):
    enc = bert_tokenizer(list(texts), truncation=True, padding='max_length',
                         max_length=BERT_MAX_LEN, return_tensors='pt')
    return enc['input_ids'], enc['attention_mask'], enc['token_type_ids']

print('Tokenizing...')
bert_train = bert_tensors(X_train_text)
bert_val = bert_tensors(X_val_text)
bert_test = bert_tensors(X_test_text)
print('BERT tensors:', tuple(bert_train[0].shape), tuple(bert_val[0].shape), tuple(bert_test[0].shape))

def bert_loader(tensors, y_t, batch_size, shuffle):
    return DataLoader(TensorDataset(*tensors, y_t), batch_size=batch_size, shuffle=shuffle)

def evaluate_bert(model, tensors, y_np, batch_size=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, mask, tt, _ in tqdm(bert_loader(tensors, torch.tensor(y_np), batch_size, False),
                                     desc='  eval', leave=False):
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                                enabled=AMP_DTYPE == torch.bfloat16):
                logits = model(ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                               token_type_ids=tt.to(DEVICE)).logits
            preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    return accuracy_score(y_np, preds), f1_score(y_np, preds, average='macro'), preds

def train_eval_bert(lr, batch_size, epochs):
    torch.manual_seed(SEED)
    model = BertForSequenceClassification.from_pretrained(BERT_NAME, num_labels=len(CLASS_NAMES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best = {'f1': -1.0, 'state': None}
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ids, mask, tt, yb in tqdm(bert_loader(bert_train, y_train_t, batch_size, True),
                                      desc=f'  epoch {epoch}/{epochs}', leave=False):
            ids, mask, tt, yb = ids.to(DEVICE), mask.to(DEVICE), tt.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE,
                                enabled=AMP_DTYPE == torch.bfloat16):
                loss = loss_fn(model(ids, attention_mask=mask, token_type_ids=tt).logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        acc, f1m, _ = evaluate_bert(model, bert_val, y_val)
        print(f'  epoch {epoch}: train_loss={total:.3f} val_acc={acc:.4f} val_f1={f1m:.4f}')
        if f1m > best['f1']:
            best = {'f1': f1m,
                    'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best['state'])
    model.to(DEVICE)
    return model, best['f1']


print(f'Training BERT: lr={LR}, batch={BATCH_SIZE}, epochs={EPOCHS} (notebook winning config)')
model, val_f1 = train_eval_bert(LR, BATCH_SIZE, EPOCHS)

acc, f1m, preds = evaluate_bert(model, bert_test, y_test)
print('\nTEST: acc=%.4f f1_macro=%.4f  (notebook reference: 0.7002 / 0.6960)' % (acc, f1m))
print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))

OUT_DIR.mkdir(parents=True, exist_ok=True)
model.config.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
model.config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
model.save_pretrained(OUT_DIR)
bert_tokenizer.save_pretrained(OUT_DIR)
print('Exported fine-tuned BERT ->', OUT_DIR.resolve())
