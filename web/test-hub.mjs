// Verifies the deployed loading path: fetch the model from the HF Hub repo
// (as the browser will) and run one prediction.
import { AutoModelForSequenceClassification, AutoTokenizer } from '@huggingface/transformers';

import { buildText, preprocessStandard } from './preprocess.js';

const REPO = 'finesse4002/yahoo-answers-bert';
const CLASS_NAMES = [
  'Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference',
  'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment',
  'Family & Relationships', 'Politics & Government',
];

const tokenizer = await AutoTokenizer.from_pretrained(REPO);
const model = await AutoModelForSequenceClassification.from_pretrained(REPO, { dtype: 'q8' });

const clean = preprocessStandard(buildText(
  'how do i fix the blue screen error on windows',
  'my pc keeps restarting with a blue screen every time i open a game',
));
const enc = await tokenizer(clean, { truncation: true, padding: 'max_length', max_length: 96 });
const { logits } = await model(enc);
const arr = Array.from(logits.data, Number);
const idx = arr.indexOf(Math.max(...arr));
console.log(`HUB LOAD OK -> ${CLASS_NAMES[idx]} (${(Math.max(...arr) > 0 ? Math.exp(Math.max(...arr)) / arr.reduce((a, b) => a + Math.exp(b), 0) * 100 : 0).toFixed(0)}% raw-max) | input: ${clean.slice(0, 50)}`);
