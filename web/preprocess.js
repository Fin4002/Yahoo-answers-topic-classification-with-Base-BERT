// Preprocessing for the web deployment - mirrors the notebook's selected
// 'lemmatized' strategy minus WordNet lemmatization, which costs nothing
// measurable on the test set (0.6993 vs 0.6990 acc, 94.4% top-1 agreement).
import { STOPWORDS } from './stopwords.js';

const URL_RE = /https?:\/\/\S+|www\.\S+/g;
const ENTITY_RE = /&\w+;/g;
const WS_RE = /\s+/g;
// Python string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
const PUNCT_RE = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g;

function preprocessMinimal(text) {
  return text
    .toLowerCase()
    .replace(URL_RE, ' ')
    .replace(ENTITY_RE, ' ')
    .replace(WS_RE, ' ')
    .trim();
}

export function preprocessStandard(text) {
  const m = preprocessMinimal(text).replace(PUNCT_RE, ' ');
  return m
    .split(' ')
    .filter((w) => w && !STOPWORDS.has(w))
    .join(' ');
}

export function buildText(title, content) {
  return [title.trim(), content.trim()].filter(Boolean).join(' ').replace(WS_RE, ' ').trim();
}
