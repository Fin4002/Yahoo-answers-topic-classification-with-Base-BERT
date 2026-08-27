# CSE440 — Lab Concepts Reference

Every concept used, demonstrated, or explained across the four lab notebooks, deduplicated and grouped by theme. Each entry is tagged with the lab(s) where it appears:

- **L1** — NLTK & ML Essentials
- **L2** — TF-IDF & Neural Networks for NLP
- **L3** — Embeddings & Sequence Models
- **L4** — Transformers (BERT) & LLMs

| Lab | Core topics | End-to-end task |
|---|---|---|
| L1 | NLTK preprocessing, BoW, classic ML | SMS spam vs. ham classification |
| L2 | TF-IDF + MLPs in Keras | Spam (binary) & coronavirus tweets (5-class sentiment) |
| L3 | Word2Vec, GloVe, RNN family | TweetEval 3-class sentiment |
| L4 | BERT, Hugging Face fine-tuning | AG News 4-class topic classification |

---

## 1. Text Processing & NLP Fundamentals

- **Corpus / corpora** (L1) — built-in NLTK corpora: `gutenberg` (Hamlet, KJV Bible), `webtext` (wine reviews); `fileids()` to enumerate documents, `.raw()` for full text.
- **Sentence tokenization** (L1) — `sent_tokenize`; demo text with "Dr.", "U.S.A.", and line breaks to show sentence-boundary ambiguity.
- **Word tokenization** (L1) — `word_tokenize`; requires downloadable Punkt tokenizer data (`punkt_tab`).
- **Stopword removal** (L1, L2) — `nltk.corpus.stopwords` manual filtering (L1); `stop_words="english"` inside vectorizers (L2). Lowercasing before the check since stopword lists are lowercase.
- **Case normalization** (L1) — `.lower()` on words, `.str.lower()` on pandas columns.
- **Non-word filtering** (L1) — `word.isalpha()` before frequency analysis.
- **Stemming** (L1) — `PorterStemmer`, `LancasterStemmer` (heavier), `SnowballStemmer`, compared side by side on morphological families (connect-, argue-, nation-, act-, derive-).
- **Lemmatization** (L1) — `WordNetLemmatizer`; POS-aware: `pos='n'` vs `pos='v'` give different lemmas.
- **Stemming vs. lemmatization** (L1) — comparative analysis table; stemming = affix stripping, lemmatization = dictionary lemma.
- **POS tagging** (L1) — `nltk.pos_tag` producing (word, tag) pairs; Penn Treebank tagset.
- **Frequency distribution** (L1) — `nltk.FreqDist`, `most_common(k)` for empirical unigram counts.
- **N-grams** (L1, L2) — `nltk.util.ngrams` (trigrams) in L1; `ngram_range=(1,2)` (uni+bigrams) as vectorizer features in L2.
- **Co-occurrence matrix** (L1) — built from scratch: V×V numpy matrix counting adjacent word pairs; vocabulary via `set()` + `sorted()`, word→index dict.
- **Tokens vs. types** (L1) — `len(tokens)` vs `len(set(tokens))`.
- **Word cloud** (L1) — `WordCloud.generate_from_frequencies(FreqDist)`, rendered with `plt.imshow`.

## 2. Text Representation / Vectorization

- **Bag-of-Words** (L1) — theory (counts per document, order ignored); implemented **from scratch** (vocab → index dict → count vectors) and with **`CountVectorizer`**; `get_feature_names_out()` to recover the vocabulary.
- **TF-IDF** (L2) — term weight = grows with in-document frequency (TF), shrinks with corpus-wide frequency (IDF); full formula shown as an image.
- **`TfidfVectorizer`** (L2) — `fit_transform` / `transform`; parameters: `max_features` (10 demo / 1000 spam / 5000 tweets), `ngram_range`, `stop_words`.
- **Sparse vs. dense matrices** (L1–2) — scipy sparse output, `.toarray()` densification; Keras `Dense` requires dense float32 arrays.
- **Fit on train only, transform test** (L1–2) — vocabulary learned from training data alone to avoid test-set leakage.
- **One-hot encoding** (L2–3) — as target encoding (`keras.utils.to_categorical` for `categorical_crossentropy`, L2); as input representation for the Word2Vec network (L3). L3 explicitly contrasts this with keeping integer labels + `sparse_categorical_crossentropy`.
- **Fixed-length text vector → MLP** (L2) — the "shallow vectorizer + Dense stack" pattern, vs. sequence input (L3) vs. contextual embeddings (L4).

## 3. Word Embeddings

- **Word embeddings** (L3) — words as dense vectors; similar meaning → similar vectors.
- **Word2Vec** (L3) — trained from scratch with gensim on a small corpus; framed as a feed-forward net: one-hot input → hidden layer (the embeddings) → context output.
- **Skip-Gram vs. CBOW** (L3) — predict context from target (`sg=1`, used) vs. target from context (`sg=0`).
- **Word2Vec hyperparameters** (L3) — `vector_size` (10), `window` (3, words each side), `min_count` frequency threshold; `build_vocab()` then `train()`.
- **gensim API** (L3) — `Word2Vec`, `model.wv['cats']` vector lookup.
- **GloVe** (L3) — embeddings from **global co-occurrence statistics** (vs. Word2Vec's local context); trained on Wikipedia/Common Crawl; pretrained 50d/100d/200d/300d; plain-text file format (word + vector per line) parsed manually.
- **Pretrained vs. learned embeddings** (L3) — loading GloVe vectors vs. a Keras `Embedding(10000, 32)` layer trained jointly with the classifier.
- **Cosine similarity** (L3) — cos(θ) = A·B / (‖A‖‖B‖); 1 = same direction, 0 = unrelated, −1 = opposite; used to show cats/dogs/mice semantic closeness; applications: text similarity, embeddings, recommenders.
- **Static vs. contextual embeddings** (L4) — Word2Vec gives "bank" one fixed vector; BERT gives a finance vs. geography vector depending on neighbors.

## 4. Machine Learning Fundamentals

- **Supervised text classification** (L1–4) — spam/ham (binary), 5-class tweet sentiment, 3-class TweetEval sentiment, 4-class AG News topics.
- **Train/test split** (L1–2) — `train_test_split(test_size=0.2, random_state=42, stratify=y)`; stratification preserves class proportions.
- **Three-way train/val/test** (L3) — using the HF dataset's native splits; validation during `fit`, test only for final evaluation.
- **Validation from training data** (L2) — `validation_split=0.2` inside Keras `fit`.
- **Reproducibility via seeds** (L1–4) — `random_state=42`, `shuffle(seed=42)`.
- **Label encoding** (L1–2) — dict `.map({'ham':0,...})`, `LabelEncoder` (with `.classes_` for report names), manual ordinal mapping for 5 sentiment classes.
- **Classic ML algorithms** (L1) — `LogisticRegression`, `MultinomialNB(alpha=1)` (**Laplace/add-one smoothing**, named explicitly), `RandomForestClassifier`; compared side by side on the same BoW features.
- **Metrics** (L1–2, L4) — accuracy, precision, recall, F1, **macro averaging**, `classification_report` (with `target_names`), `confusion_matrix`; Keras `metrics=['accuracy']` (+'precision').
- **Overfitting** (L2–3) — diagnosed from train-vs-validation loss/accuracy divergence; motivates dropout and early stopping.
- **Regularization: Dropout** (L2–3) — `Dropout(0.2/0.3/0.5)` after hidden/recurrent layers.
- **EarlyStopping callback** (L2) — `monitor="val_loss", patience=3, restore_best_weights=True`.
- **Class balance inspection** (L2) — `df.value_counts("Sentiment")`.
- **Inference helper pattern** (L1, L4) — reusable `predict_spam()` / `predict()` chaining preprocess → vectorize/tokenize → model → decode label.

## 5. Neural Networks (Keras / TensorFlow)

- **Multilayer perceptron** (L2–3) — stacks of `Dense` layers; architecture diagram shown; depths varied (32→16, 32→16→8, 256→128).
- **Input layer** (L2) — `Input(shape=(n_features,))`.
- **Activation functions** (L2) — `relu`, `leaky_relu` (hidden); `sigmoid` (binary output), `softmax` (multi-class output); `tanh`, `linear` listed as options.
- **Loss functions** (L2–3) — `binary_crossentropy`, `categorical_crossentropy` (one-hot targets), `sparse_categorical_crossentropy` (integer targets).
- **Optimizers** (L2–3) — `adam` used throughout; `sgd`, `rmsprop`, `adagrad`, `adadelta` listed.
- **Training loop mechanics** (L2–3) — epochs, `batch_size=64` (mini-batch gradient descent), `verbose` levels.
- **Backpropagation & gradient descent** (L3) — explained as the mechanism adjusting Word2Vec weights to minimize prediction error.
- **Training history** (L2) — `history.history` dict (`loss`, `val_loss`, `accuracy`, `val_accuracy`) → train-vs-validation loss and accuracy curves per epoch (matplotlib).
- **Probability decoding** (L2) — sigmoid thresholding `(p >= 0.5).astype(int)`; softmax `np.argmax(..., axis=1)`; `.ravel()` to flatten.
- **Keras workflow** (L2–3) — `Sequential` → `compile(optimizer, loss, metrics)` → `summary()` → `fit` → `evaluate` → `predict`.

## 6. Sequence Models

- **Text → integer sequences** (L3) — `Tokenizer(num_words=10000, oov_token="<OOV>")`, `fit_on_texts`, `texts_to_sequences`; index conventions 0=PAD, 1=OOV, 2+=words by frequency.
- **Padding** (L3) — `pad_sequences(maxlen=50, padding="post")`; fixed-length input requirement of neural nets; post-padding = zeros appended at the end.
- **Recurrent architectures** (L3) — `SimpleRNN(32)`, `LSTM(32)`, `GRU(32)` — same pipeline run across all variants for accuracy comparison.
- **Bidirectionality** (L3) — `Bidirectional(LSTM(32))`, `Bidirectional(GRU(32))` processing the sequence both directions.
- **Embedding layer in a classifier** (L3) — `Embedding(10000, 32, input_length=50)` feeding the recurrent layer.
- **GPU placement** (L3) — `with tf.device('/GPU:0'):` around build/train.

## 7. Transformers, BERT & LLMs

- **The original Transformer** (L4) — 2017 "Attention Is All You Need" encoder–decoder; BERT and GPT are its two halves.
- **Encoder-only (BERT)** (L4) — stacked encoders, bidirectional context (left + right); rich per-token vectors; NLU: classification, NER, search.
- **Decoder-only (GPT)** (L4) — unidirectional (left-only); generates tokens one at a time; NLG: chat, translation, code, summarization. Modern examples: GPT-3.5/4, Claude, Llama.
- **Masked Language Modeling** (L4) — randomly mask 15% of tokens and predict them; **80/10/10 rule**: of the chosen tokens, 80% → `[MASK]`, 10% → random token, 10% unchanged.
- **Next Sentence Prediction** (L4) — binary IsNext/NotNext over 50% true / 50% random sentence pairs; input format `[CLS] A [SEP] B [SEP]`; motivates QA/NLI.
- **Special tokens** (L4) — `[CLS]`, `[SEP]`, `[MASK]`.
- **Self-supervised pre-training** (L4) — no human labels; English Wikipedia (2.5B words) + BookCorpus (800M words); compute cost: BERT-base ≈ 4 days on 16 TPUs, BERT-large on 64.
- **Transfer learning: pre-train → fine-tune** (L4) — task-agnostic foundation model first, then fine-tune a copy on a small labeled dataset in minutes–hours.
- **Classification head** (L4) — `AutoModelForSequenceClassification` with `num_labels=4` on top of the pretrained backbone; full end-to-end fine-tuning (no layer freezing).
- **BERT tokenizer outputs** (L4) — `input_ids` (vocab IDs), `token_type_ids` (segment membership), `attention_mask` (1=real token, 0=pad); subword pieces inspected via `convert_ids_to_tokens` (WordPiece, implicit).
- **Tokenizer options** (L4) — `truncation=True, max_length=128`, `padding=True`, `return_tensors="pt"`.
- **Hugging Face `Trainer` stack** (L4) — `Trainer` + `TrainingArguments` (`num_train_epochs=2`, `per_device_train_batch_size=16`, `logging_steps=50`, `eval_strategy="epoch"`, `output_dir`) + `compute_metrics` (accuracy from logits via argmax).
- **`DataCollatorWithPadding`** (L4) — dynamic per-batch padding instead of padding the whole dataset statically.
- **`datasets` library** (L3–4) — `load_dataset` (`cardiffnlp/tweet_eval`, `fancyzhx/ag_news`), `Dataset.map(tokenize, batched=True)`, `shuffle(seed=42).select(range(n))` subsampling, `rename_column("label", "labels")` for Trainer, `features["label"].names` class-name introspection.
- **PyTorch inference** (L4) — `torch.no_grad()`, `outputs.last_hidden_state` for contextual embeddings, logits + argmax decode, `.to(model.device)` device placement.
- **NTP & prompting** (L4) — next-token prediction as the decoder-only objective; decoder-only models adapted via prompts (no prompt for encoders).
- **Model scaling dimensions** (L4) — layers / attention heads / hidden size / parameter count (BERT-base: 12/12/768/110M; BERT-large: 24/16/1024/340M).
- **BERT variant landscape** (L4) — **DistilBERT** (knowledge distillation, 6 layers, 66M), **RoBERTa** (better training recipe, MLM-only, no NSP), **ALBERT** (cross-layer parameter sharing, SOP instead of NSP), **ELECTRA** (replaced-token detection), **DeBERTa-v3** (disentangled content/position attention); accuracy-vs-efficiency trade-offs per variant.

## 8. Libraries & APIs Used

| Library | Labs | Key APIs |
|---|---|---|
| nltk | L1 | `corpus` (gutenberg, webtext, stopwords), `sent_tokenize`, `word_tokenize`, `pos_tag`, `PorterStemmer`, `LancasterStemmer`, `SnowballStemmer`, `WordNetLemmatizer`, `FreqDist`, `ngrams` |
| scikit-learn | L1–2, L4 | `CountVectorizer`, `TfidfVectorizer`, `train_test_split`, `LogisticRegression`, `MultinomialNB`, `RandomForestClassifier`, `LabelEncoder`, `classification_report`, `confusion_matrix`, `accuracy_score`, `f1_score`, `cosine_similarity` |
| tensorflow / keras | L2–3 | `Sequential`, `Input`, `Dense`, `Dropout`, `Embedding`, `SimpleRNN`, `LSTM`, `GRU`, `Bidirectional`, `EarlyStopping`, `Tokenizer`, `pad_sequences`, `to_categorical`, `compile/fit/evaluate/predict/summary`, `tf.device` |
| gensim | L3 | `Word2Vec` (`build_vocab`, `train`, `wv`) |
| transformers | L4 | `AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`, `DataCollatorWithPadding` |
| datasets | L3–4 | `load_dataset`, `Dataset.map/shuffle/select/rename_column` |
| torch | L4 | `no_grad`, tensor ops, device movement |
| pandas | L1–2 | `read_csv`, `head`, `value_counts`, `drop`, `map`, `astype`, DataFrame as visualization |
| numpy | L1–4 | `zeros`, `argmax`, array casting (`astype("float32")`), `ravel` |
| matplotlib | L1–2 | `plot`, `bar`, `imshow`, titles/labels/legend/grid, train-vs-val curves |
| wordcloud | L1 | `WordCloud`, `generate_from_frequencies` |
| tabulate | L1 | grid tables for stemmer comparisons |

## 9. Tooling & Environment

- **Google Colab** (L1–3) — notebook environment; `!pip install` shell escapes.
- **Google Drive data loading** (L1–3) — parsing share links (`split("/")[-2]`) → direct-download URL `drive.google.com/uc?id=...`; `drive.mount()` for local files (GloVe).
- **NLTK data downloads** (L1) — `nltk.download(...)` for corpora and tokenizer models.
- **From-scratch vs. library implementations** (L1) — BoW built manually, then via `CountVectorizer`.
- **EDA before modeling** (L1–3) — `.head()`, `.shape`, class distributions, inspecting samples at each pipeline stage.
- **Sanity-check pattern** (L3) — printing `dataset["train"][0]`, `X_train[0]`, `y_train` after each transformation.
- **Python idioms throughout** — list comprehensions, dict comprehensions as index maps, `enumerate`, `set`/`sorted` for vocabularies, tuple unpacking, f-strings, docstringed helper functions, string methods (`.lower()`, `.isalpha()`, `.split()`).
- **Naming conventions** — `random_state=42` / `seed=42` for reproducibility.

## 10. Datasets Used Across Labs

| Dataset | Lab | Task |
|---|---|---|
| SMS Spam Collection (Category/Message) | L1, L2 | Binary ham vs. spam |
| Kaggle Coronavirus Tweets | L2 | 5-class sentiment (Extremely Negative → Extremely Positive) |
| `cardiffnlp/tweet_eval` (sentiment) | L3 | 3-class sentiment (neg/neu/pos) |
| `fancyzhx/ag_news` | L4 | 4-class news topic (World/Sports/Business/Sci-Tech) |
| NLTK Gutenberg / WebText corpora | L1 | Preprocessing demos |
