// Node end-to-end test of the web deployment path: same library
// (@huggingface/transformers), same model files, same preprocessing.
import { AutoModelForSequenceClassification, AutoTokenizer } from '@huggingface/transformers';

import { buildText, preprocessStandard } from './preprocess.js';

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

const CASES = [
  ['how do i fix the blue screen error on windows', 'my pc keeps restarting with a blue screen every time i open a game what should i do', 'Computers & Internet'],
  ['best way to learn calculus', 'i am struggling with derivatives and integrals in my first year course any advice', 'Science & Mathematics'],
  ['sore throat for a week', 'i have had a sore throat and mild fever for 7 days should i see a doctor', 'Health'],
  ['who will win the election next year', 'do you think the current administration policies will help them win reelection', 'Politics & Government'],
  ['1990s cartoon with a talking dog', 'trying to remember an old cartoon from the 90s where the dog solves mysteries', 'Entertainment'],
];

const tokenizer = await AutoTokenizer.from_pretrained('./model');
const model = await AutoModelForSequenceClassification.from_pretrained('./model', { dtype: 'q8' });

const softmax = (xs) => {
  const m = Math.max(...xs);
  const es = xs.map((x) => Math.exp(x - m));
  const s = es.reduce((a, b) => a + b, 0);
  return es.map((e) => e / s);
};

let correct = 0;
for (const [title, content, expected] of CASES) {
  const clean = preprocessStandard(buildText(title, content));
  const enc = await tokenizer(clean, { truncation: true, padding: 'max_length', max_length: 96 });
  const { logits } = await model(enc);
  const probs = softmax(Array.from(logits.data, Number));
  const idx = probs.indexOf(Math.max(...probs));
  const ok = CLASS_NAMES[idx] === expected;
  if (ok) correct += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${(probs[idx] * 100).toFixed(1)}%  ${CLASS_NAMES[idx].padEnd(26)} (expected ${expected}) | preprocessed: ${clean.slice(0, 60)}`);
}
console.log(`\n${correct}/${CASES.length} correct`);
