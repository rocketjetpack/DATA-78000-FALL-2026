#!/usr/bin/env python3

import os
import re
import sys
import pickle
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaMulticore, Phrases
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import pos_tag
import nltk
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'gc_dissertations_combined_v2.csv'
OUTPUT_DIR = SCRIPT_DIR / 'output'
TEXT_COLUMN = 'abstract'

# Settings pulled from the collab code
# np.random.seed is not a reliable way to handle randomness in Python/Numpy
rnd_gen = np.random.default_rng(seed=2026)

# data settings
USE_SUBSET = False
DO_TESTING = False
TRUNCATE_SIZE = 2000  # For testing
MIN_ABSTRACT_WORDS = 20  # drop near-empty/placeholder abstracts

# model settings
# LdaMulticore actually uses the worker count, unlike the skeleton's declared-but-unused
# WORKERS=8. Preprocessing (POS tagging + lemmatizing) is CPU-bound and embarrassingly
# parallel across documents, so it runs in a worker pool too.
WORKERS = max(1, (os.cpu_count() or 2) - 1)
NUM_TOPICS = 20
PASSES = 10

RUN_COHERENCE_SWEEP = False  # topic-count selection; slow, run once then set NUM_TOPICS
NUM_TOPICS_RANGE = range(5, 31, 5)

# Preprocessing sensitivity check (Denny & Spirling 2018): preprocessing choices form a
# garden of forking paths, so rather than silently picking one, we train the same model
# under several and compare. PRIMARY_VARIANT feeds the downstream advisor analysis.
PRIMARY_VARIANT = 'nouns'
RUN_ALL_VARIANTS = True

# dictionary settings
MAX_WORDS = 100000
MIN_WORDS = 15  # drop terms in fewer than N documents (typos/idiosyncratic terms)

VARIANTS = {
	# Reproduces the Colab/skeleton pattern, kept only as the sensitivity baseline.
	# no_above=0.5 is effectively inert on this corpus: the most common content word
	# ("study") appears in 49.2% of abstracts, so nothing reaches the ceiling.
	'baseline': dict(stem=True, lemma_pos='v', nouns_only=False, bigrams=False,
		min_token_len=4, no_above=0.5),
	# Recommended. No stemming (Schofield & Mimno 2016 find stemmers don't improve
	# topic models and produce unreadable tokens like "signific"/"studi"); nouns only,
	# because the boilerplate crowding this corpus ("examines", "significant",
	# "different") is overwhelmingly verbs and adjectives, while nouns carry subject
	# matter; bigrams for readability; corpus-adaptive frequency ceiling.
	'nouns': dict(stem=False, lemma_pos='n', nouns_only=True, bigrams=True,
		min_token_len=3, no_above=0.15),
	# Isolates what the noun filter is responsible for -- identical to 'nouns' otherwise.
	'all_pos': dict(stem=False, lemma_pos='n', nouns_only=False, bigrams=True,
		min_token_len=3, no_above=0.15),
}

# Domain stopwords. Derived by ranking document frequency across the corpus rather than
# brainstormed: these are the terms that appear in a large share of abstracts regardless
# of subject, so they cannot discriminate topics. The no_above ceiling catches most of
# them automatically; this list is the belt-and-braces supplement.
EXTRA_STOPWORDS = frozenset({
	'abstract', 'dissertation', 'thesis', 'study', 'studies', 'research', 'chapter',
	'university', 'graduate', 'center', 'cuny', 'author', 'copyright', 'page',
	'analysis', 'result', 'results', 'finding', 'findings', 'data', 'work',
	'approach', 'method', 'methods', 'literature', 'question', 'questions',
	'example', 'examples', 'case', 'cases', 'term', 'terms', 'part', 'number',
})
STOPWORD_SET = STOPWORDS | EXTRA_STOPWORDS

# Citation/numbering debris seen in the document-frequency tail (roman numerals survive
# the length filter and are frequent enough to survive the no_below floor).
JUNK_TOKENS = frozenset({
	'iii', 'iiii', 'vii', 'viii', 'xii', 'xiii', 'xiv', 'xvi', 'xvii', 'xviii',
	'ibid', 'etc', 'pp', 'vol', 'nos',
})

# ProQuest escapes "$" as "{dollar}", so LaTeX math arrives as literal prose:
#   "{dollar}\alpha{dollar}-factor ... OCH{dollar}\sb3{dollar}"
# 217 abstracts contain 20+ of these, concentrated in Chemistry/Physics/Biology/Math.
# Left uncleaned, "dollar" (and the \sb/\sp subscript markers) become a proxy for
# "STEM dissertation that used equations" and the model will happily treat that
# markup artifact as topical content. Strip the math spans rather than stopwording
# the word "dollar", which is legitimate content in the Economics abstracts.
MATH_SPAN_RE = re.compile(r'\{dollar\}.*?\{dollar\}', re.S)
LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+')


def clean_markup(text):
	text = MATH_SPAN_RE.sub(' ', text)
	text = text.replace('{dollar}', ' ')  # unbalanced stragglers
	return LATEX_CMD_RE.sub(' ', text)

_worker_config = None
_stemmer = None
_lemmatizer = None


def _init_worker(config):
	global _worker_config
	_worker_config = config


def _get_text_tools():
	# Recreated per worker process rather than shared, since these aren't safely
	# shareable across a multiprocessing.Pool.
	global _stemmer, _lemmatizer
	if _lemmatizer is None:
		_stemmer = SnowballStemmer('english')
		_lemmatizer = WordNetLemmatizer()
	return _stemmer, _lemmatizer


def preprocess(text):
	if pd.isna(text) or not isinstance(text, str):
		return []

	config = _worker_config
	stemmer, lemmatizer = _get_text_tools()
	tokens = gensim.utils.simple_preprocess(clean_markup(text), deacc=True)

	# POS tagging runs on the full token sequence before any filtering, because the
	# tagger needs surrounding context to disambiguate; filtering first would corrupt it.
	if config['nouns_only']:
		tokens = [tok for tok, tag in pos_tag(tokens) if tag.startswith('NN')]

	result = []
	for token in tokens:
		# Length filter is `< min_token_len`, not the skeleton's `<= 3`, which silently
		# deleted art/war/law/sex/men/dna/ion/job -- some of the most discriminative
		# terms in a corpus spanning Art History through Chemistry.
		if len(token) < config['min_token_len']:
			continue
		if token in STOPWORD_SET or token in JUNK_TOKENS:
			continue
		lemma = lemmatizer.lemmatize(token, pos=config['lemma_pos'])
		# Stopwords are checked on the surface form above; stemming/lemmatizing creates
		# new forms ("significant" -> "signific") that no stoplist contains, so check again.
		if lemma in STOPWORD_SET:
			continue
		result.append(stemmer.stem(lemma) if config['stem'] else lemma)
	return result


def preprocess_corpus(texts, config):
	with mp.Pool(processes=WORKERS, initializer=_init_worker, initargs=(config,)) as pool:
		tokenized = pool.map(preprocess, texts, chunksize=200)

	if config['bigrams']:
		# Phrases must see the whole corpus, so it's trained in the parent after the
		# parallel tokenization pass. Turns "public health" into "public_health".
		phrases = Phrases(tokenized, min_count=20, threshold=10.0)
		tokenized = [phrases[doc] for doc in tokenized]
	return tokenized


def build_corpus(tokenized, no_above):
	dictionary = corpora.Dictionary(tokenized)
	dictionary.filter_extremes(no_below=MIN_WORDS, no_above=no_above, keep_n=MAX_WORDS)
	corpus = [dictionary.doc2bow(toks) for toks in tokenized]
	return dictionary, corpus


def train(dictionary, corpus, num_topics):
	return LdaMulticore(
		corpus=corpus, id2word=dictionary, num_topics=num_topics,
		passes=PASSES, workers=WORKERS, random_state=2026,
	)


def report_cross_topic_terms(model, num_topics, topn=20):
	"""Terms in the top-N of many topics carry no discriminating power.

	This is the empirical way to grow EXTRA_STOPWORDS -- measured from the fitted
	model rather than guessed. Printed for inspection, deliberately not auto-applied.
	"""
	counts = Counter()
	for topic_id in range(num_topics):
		for term, _ in model.show_topic(topic_id, topn=topn):
			counts[term] += 1
	flagged = [(t, c) for t, c in counts.most_common() if c >= max(3, num_topics // 4)]
	if flagged:
		print(f'  Cross-topic terms (in top-{topn} of >= {max(3, num_topics // 4)} topics) '
			f'-- candidates for EXTRA_STOPWORDS:')
		print('   ', ', '.join(f'{t}({c})' for t, c in flagged[:25]))
	else:
		print(f'  No term appears in the top-{topn} of many topics.')


def sweep_coherence(dictionary, corpus, texts):
	scores = []
	for k in NUM_TOPICS_RANGE:
		model = train(dictionary, corpus, k)
		coherence = CoherenceModel(
			model=model, texts=texts, dictionary=dictionary, coherence='c_v',
		).get_coherence()
		print(f'  num_topics={k}: coherence={coherence:.4f}')
		scores.append((k, coherence))

	ks, values = zip(*scores)
	plt.figure(figsize=(8, 5))
	plt.plot(ks, values, marker='o')
	plt.xlabel('Number of topics')
	plt.ylabel('Coherence (c_v)')
	plt.title('Topic count vs. coherence')
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / 'coherence_scores.png')
	plt.close()
	return scores


def write_document_topics(model, corpus, record_ids, num_topics, path):
	rows = []
	for record_id, bow in zip(record_ids, corpus):
		dist = dict(model.get_document_topics(bow, minimum_probability=0.0))
		dominant_topic = max(dist, key=dist.get)
		row = {
			'record_id': record_id,
			'dominant_topic': dominant_topic,
			'dominant_topic_prob': dist[dominant_topic],
		}
		row.update({f'topic_{t}': dist.get(t, 0.0) for t in range(num_topics)})
		rows.append(row)
	pd.DataFrame(rows).to_csv(path, index=False)
	return len(rows)


def run_variant(name, config, usable):
	print(f'\n=== VARIANT: {name} ===')
	print(f'  {config}')
	tokenized = preprocess_corpus(usable[TEXT_COLUMN].tolist(), config)

	keep = [i for i, toks in enumerate(tokenized) if toks]
	rows = usable.iloc[keep].reset_index(drop=True)
	texts = [tokenized[i] for i in keep]
	print(f'  Documents surviving preprocessing: {len(texts)} of {len(usable)}')

	dictionary, corpus = build_corpus(texts, config['no_above'])
	print(f'  Vocabulary after filter_extremes(no_below={MIN_WORDS}, '
		f'no_above={config["no_above"]}): {len(dictionary)} terms')
	print(f'  Mean tokens/doc after filtering: {np.mean([len(d) for d in corpus]):.1f}')

	if RUN_COHERENCE_SWEEP and name == PRIMARY_VARIANT:
		print('  Sweeping topic counts for coherence...')
		sweep_coherence(dictionary, corpus, texts)

	model = train(dictionary, corpus, NUM_TOPICS)
	coherence = CoherenceModel(
		model=model, texts=texts, dictionary=dictionary, coherence='c_v',
	).get_coherence()
	print(f'  Coherence (c_v) at num_topics={NUM_TOPICS}: {coherence:.4f}')

	for topic_id, words in model.print_topics(num_topics=NUM_TOPICS, num_words=8):
		print(f'    Topic {topic_id}: {words}')
	report_cross_topic_terms(model, NUM_TOPICS)

	dictionary.save(str(OUTPUT_DIR / f'dictionary_{name}.dict'))
	model.save(str(OUTPUT_DIR / f'lda_model_{name}.model'))
	n = write_document_topics(model, corpus, rows['record_id'], NUM_TOPICS,
		OUTPUT_DIR / f'document_topics_{name}.csv')

	if name == PRIMARY_VARIANT:
		# Downstream scripts read this stable filename.
		write_document_topics(model, corpus, rows['record_id'], NUM_TOPICS,
			OUTPUT_DIR / 'document_topics.csv')
		with open(OUTPUT_DIR / 'corpus.pkl', 'wb') as f:
			pickle.dump(corpus, f)

	print(f'  Wrote {n} document-topic rows.')
	return {'variant': name, 'coherence': coherence, 'vocab': len(dictionary),
		'documents': len(texts)}


def main():
	# Guarded by `if __name__ == '__main__'` below: on macOS/Windows, multiprocessing's
	# spawn start method re-imports this module in every worker process, so anything
	# with side effects (path checks, downloads, mkdir) must live in main() rather than
	# at module level, or it silently re-runs once per worker.
	print('Data source: ', DATA_PATH)
	if not DATA_PATH.exists():
		print('Data source is not available or does not exist.')
		sys.exit(1)
	OUTPUT_DIR.mkdir(exist_ok=True)
	for resource in ('wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng'):
		nltk.download(resource, quiet=True)

	data = pd.read_csv(DATA_PATH, low_memory=False)
	print(data.shape)

	usable = data[
		data[TEXT_COLUMN].notna()
		& (data['abstract_is_placeholder'] == 0)
		& (data['abstract_wordcount'] >= MIN_ABSTRACT_WORDS)
	].reset_index(drop=True)
	print(f'Usable abstracts: {len(usable)} of {len(data)}')
	# Real abstracts are ~0% before 1980 and ~99% after, so the topic model necessarily
	# describes 1980-2026 even though the corpus starts in 1965.
	print(f'Year range of modeled abstracts: {int(usable["year"].min())}-{int(usable["year"].max())}')

	if USE_SUBSET and DO_TESTING:
		# Random, not the first N rows: the CSV is in chronological order, so a head
		# slice would train only on the oldest abstracts and misrepresent the corpus.
		usable = usable.sample(TRUNCATE_SIZE, random_state=2026).reset_index(drop=True)
		print(f'Testing on random subset of {len(usable)} rows')

	names = list(VARIANTS) if RUN_ALL_VARIANTS else [PRIMARY_VARIANT]
	summary = [run_variant(name, VARIANTS[name], usable) for name in names]

	print('\n=== PREPROCESSING SENSITIVITY ===')
	summary_df = pd.DataFrame(summary)
	print(summary_df.to_string(index=False))
	summary_df.to_csv(OUTPUT_DIR / 'preprocessing_sensitivity.csv', index=False)
	print('\nCoherence is comparable across variants only as a rough signal -- c_v is not '
		'a ground truth, and Chang et al. (2009) found fit metrics can correlate '
		'negatively with human interpretability. Read the topic words.')


if __name__ == '__main__':
	main()
