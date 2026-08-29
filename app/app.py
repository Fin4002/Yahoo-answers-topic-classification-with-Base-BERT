"""
Gradio interface for the fine-tuned BERT Base classifier.

Serves the 10-class Yahoo! Answers topic classifier trained in
notebooks/yahoo_answers_topic_classification_full.ipynb (Section 9, BERT Base,
best config lr=2e-5, batch=32, epochs=3 - 70.02% test acc / 0.696 macro-F1).

The prediction pipeline replicates the notebook exactly:
  question_title + ' ' + question_content  (build_text)
  -> preprocess_lemmatized (the strategy selected in the notebook)
  -> bert-base-uncased tokenizer, max_len=96
  -> BertForSequenceClassification -> softmax over the 10 topics

Model resolution order:
  1. ./model            (created by the export cell - see export_cell.txt)
  2. MODEL_ID env var   (HF Hub repo id, used when hosted on Spaces)

Run locally:  python app.py   (from the app/ folder)
"""

import os
import re
import string
import sys
from pathlib import Path

import gradio as gr
import nltk
import torch
from nltk.corpus import stopwords as nltk_stopwords
from transformers import AutoTokenizer, BertForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_ID = os.environ.get('MODEL_ID', '').strip()

BERT_NAME = 'bert-base-uncased'
MAX_LEN = 96

CLASS_NAMES = [
    'Society & Culture',
    'Science & Mathematics',
    'Health',
    'Education & Reference',
    'Computers & Internet',
    'Sports',
    'Business & Finance',
    'Entertainment',
    'Family & Relationships',
    'Politics & Government',
]

# ZeroGPU Spaces attach a GPU only inside @spaces.GPU calls - moving the model
# to CUDA at startup there crashes, so stay on CPU (inference is fast enough).
if os.environ.get('SPACES_ZERO_GPU', '') == '1':
    DEVICE = torch.device('cpu')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device('xpu')
else:
    DEVICE = torch.device('cpu')

for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as exc:
        print(f'NLTK download failed for {resource}: {exc}')

LEMMATIZER = nltk.stem.WordNetLemmatizer()
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


def preprocess_lemmatized(text):
    tokens = [LEMMATIZER.lemmatize(w) for w in preprocess_standard(text).split()]
    return ' '.join(tokens)


def build_text(title, content):
    merged = ' '.join(part for part in (title.strip(), content.strip()) if part)
    return re.sub(r'\s+', ' ', merged).strip()


def load_model():
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        src = str(MODEL_DIR)
        print(f'Loading fine-tuned model from {src}')
    elif MODEL_ID:
        src = MODEL_ID
        print(f'Loading fine-tuned model from the HF Hub: {src}')
    else:
        sys.exit(
            'No model found. Either:\n'
            f'  1. export the trained BERT into {MODEL_DIR} - run the cell in\n'
            '     app/export_cell.txt at the end of the notebook, or\n'
            '  2. set the MODEL_ID env var to a Hugging Face repo id.'
        )
    tokenizer = AutoTokenizer.from_pretrained(src)
    model = BertForSequenceClassification.from_pretrained(src)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

id2label = getattr(model.config, 'id2label', None)
if id2label and id2label.get(0) not in (None, 'LABEL_0'):
    CLASS_NAMES = [id2label[i] for i in range(len(CLASS_NAMES))]


def predict(title, content):
    text = build_text(title or '', content or '')
    if not text:
        return {}, 'Enter a question (title and/or content) to get a prediction.'

    clean = preprocess_lemmatized(text)
    enc = tokenizer(clean, truncation=True, padding='max_length',
                    max_length=MAX_LEN, return_tensors='pt')
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()

    confidences = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
    top = max(confidences, key=confidences.get)
    return confidences, f'Preprocessed input: {clean}\n\nPredicted topic: {top}'


EXAMPLES = [
    ['how do i fix the blue screen error on windows', 'my pc keeps restarting with a blue screen every time i open a game what should i do'],
    ['best way to learn calculus', 'i am struggling with derivatives and integrals in my first year course any advice'],
    [' sore throat for a week', 'i have had a sore throat and mild fever for 7 days should i see a doctor'],
    ['who will win the election next year', 'do you think the current administration policies will help them win reelection'],
    ['1990s cartoon with a talking dog', 'trying to remember an old cartoon from the 90s where the dog solves mysteries'],
]


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label='Question title', placeholder='e.g. how do i fix the blue screen error on windows'),
        gr.Textbox(label='Question content', lines=5, placeholder='The body of the question...'),
    ],
    outputs=[
        gr.Label(label='Topic probabilities', num_top_classes=10),
        gr.Textbox(label='Model input (preprocessed)', lines=3, interactive=False),
    ],
    title='Yahoo! Answers Topic Classification - Base BERT',
    description=(
        'Fine-tuned bert-base-uncased classifying a question into one of 10 topics. '
        'Trained/evaluated in the project notebook: **70.02% test accuracy / 0.696 macro-F1** '
        '(best of 11 models). CSE440 lab project.'
    ),
    examples=EXAMPLES,
)

if __name__ == '__main__':
    demo.launch()
