// Web deployment: fine-tuned BERT classifier running fully client-side via
// transformers.js (onnxruntime-web, int8 quantized). Mirrors app/app.py:
// buildText -> preprocessStandard -> tokenizer(max_len=96) -> softmax.
import { AutoModelForSequenceClassification, AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';

import { buildText, preprocessStandard } from './preprocess.js';

// Weights live in a public HF model repo (105 MB exceeds Vercel's 100 MB
// per-file limit); the browser fetches them from the Hub CDN on first visit.
const HF_REPO = 'finesse4002/yahoo-answers-bert';

env.allowLocalModels = false;
env.allowRemoteModels = true;

const CLASS_NAMES = [
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
];
const MAX_LEN = 96;

const EXAMPLES = [
  ['how do i fix the blue screen error on windows', 'my pc keeps restarting with a blue screen every time i open a game what should i do'],
  ['best way to learn calculus', 'i am struggling with derivatives and integrals in my first year course any advice'],
  ['sore throat for a week', 'i have had a sore throat and mild fever for 7 days should i see a doctor'],
  ['who will win the election next year', 'do you think the current administration policies will help them win reelection'],
  ['1990s cartoon with a talking dog', 'trying to remember an old cartoon from the 90s where the dog solves mysteries'],
];

const $ = (id) => document.getElementById(id);
const classifyBtn = $('classify');

function showError(msg) {
  const el = $('error');
  el.textContent = msg;
  el.hidden = false;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

async function init() {
  const loadCard = $('loading');
  loadCard.hidden = false;
  const onProgress = (p) => {
    if (p.status === 'progress' && p.file && p.file.includes('model_quantized')) {
      $('load-status').textContent = `Downloading model (${(p.loaded / 1e6).toFixed(0)} / ${(p.total / 1e6).toFixed(0)} MB)…`;
      $('load-fill').style.width = `${p.progress?.toFixed(1) ?? 0}%`;
    } else if (p.status === 'ready') {
      $('load-status').textContent = 'Model ready.';
      $('load-fill').style.width = '100%';
    }
  };

  try {
    const tokenizer = await AutoTokenizer.from_pretrained(HF_REPO);
    const model = await AutoModelForSequenceClassification.from_pretrained(HF_REPO, {
      dtype: 'q8',
      progress_callback: onProgress,
    });
    window.__classify = async (title, content) => {
      const text = buildText(title, content);
      if (!text) return null;
      const clean = preprocessStandard(text);
      const enc = await tokenizer(clean, {
        truncation: true,
        padding: 'max_length',
        max_length: MAX_LEN,
      });
      const { logits } = await model(enc);
      const probs = softmax(Array.from(logits.data, Number));
      return { probs, clean };
    };
    classifyBtn.disabled = false;
    setTimeout(() => (loadCard.hidden = true), 600);
  } catch (err) {
    showError(`Failed to load the model: ${err}. Check the network connection and reload.`);
  }
}

function render({ probs, clean }) {
  const rows = CLASS_NAMES.map((name, i) => ({ name, p: probs[i] })).sort((a, b) => b.p - a.p);
  const top = rows[0];

  $('topic').textContent = top.name;
  $('conf').textContent = `${(top.p * 100).toFixed(1)}% confidence`;
  $('preprocessed').textContent = clean || '(empty)';

  const bars = $('bars');
  bars.textContent = '';
  for (const r of rows) {
    const row = document.createElement('div');
    row.className = 'bar-row' + (r === top ? ' winner' : '');

    const label = document.createElement('span');
    label.className = 'bar-label';
    label.textContent = r.name;

    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = `${Math.max(r.p * 100, 1).toFixed(2)}%`;
    track.appendChild(fill);

    const value = document.createElement('span');
    value.className = 'bar-value';
    value.textContent = `${(r.p * 100).toFixed(1)}%`;

    row.append(label, track, value);
    bars.appendChild(row);
  }
  $('result').hidden = false;
}

async function onClassify() {
  if (!window.__classify) return;
  classifyBtn.disabled = true;
  classifyBtn.textContent = 'Classifying…';
  try {
    const out = await window.__classify($('q-title').value, $('q-content').value);
    if (!out) {
      showError('Enter a question (title and/or content) first.');
    } else {
      $('error').hidden = true;
      render(out);
    }
  } catch (err) {
    showError(`Prediction failed: ${err}`);
  } finally {
    classifyBtn.disabled = false;
    classifyBtn.textContent = 'Classify';
  }
}

function mountExamples() {
  const wrap = $('examples');
  const short = (t) => (t.length > 34 ? t.slice(0, 33) + '…' : t);
  EXAMPLES.forEach(([title, content]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = short(title);
    b.addEventListener('click', () => {
      $('q-title').value = title;
      $('q-content').value = content;
      onClassify();
    });
    wrap.appendChild(b);
  });
}

classifyBtn.addEventListener('click', onClassify);
mountExamples();
init();
