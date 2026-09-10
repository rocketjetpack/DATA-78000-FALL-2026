#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from common import people_on


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'gc_dissertations_combined_v2.csv'
OUTPUT_DIR = SCRIPT_DIR / 'output'
DOC_TOPICS_PATH = OUTPUT_DIR / 'document_topics.csv'
GRAPH_PATH = OUTPUT_DIR / 'advisor_network.graphml'

for path in (DATA_PATH, DOC_TOPICS_PATH, GRAPH_PATH):
	if not path.exists():
		print(f'Required input missing: {path}')
		print('Run topic_model.py and advisor_network.py first.')
		sys.exit(1)

MIN_DISSERTATIONS = 5  # below this an advisor's topic profile is mostly LDA noise:
# abstracts are short, per-document topic distributions are unstable point estimates,
# and averaging 3 of them does not stabilise them
N_PERMUTATIONS = 1000
rng = np.random.default_rng(seed=2026)


def jensen_shannon(P, i, j):
	"""Mean Jensen-Shannon divergence (base 2, in [0,1]) over the node pairs (i, j)."""
	p, q = P[i], P[j]
	m = 0.5 * (p + q)
	with np.errstate(divide='ignore', invalid='ignore'):
		kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0).sum(axis=1)
		kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0).sum(axis=1)
	return 0.5 * kl_pm + 0.5 * kl_qm


def permutation_test(P, i, j, label):
	"""Are connected people more topically similar than the same graph would give by chance?

	The null shuffles topic profiles across nodes while holding the edge list fixed,
	so it preserves the network's structure exactly and asks only whether *who* holds
	which profile matters. Lower JSD = more similar.
	"""
	observed = jensen_shannon(P, i, j).mean()
	null = np.empty(N_PERMUTATIONS)
	n = P.shape[0]
	for k in range(N_PERMUTATIONS):
		null[k] = jensen_shannon(P[rng.permutation(n)], i, j).mean()

	# One-sided: we're asking whether observed similarity is HIGHER (divergence lower).
	p_value = (np.sum(null <= observed) + 1) / (N_PERMUTATIONS + 1)
	z = (observed - null.mean()) / null.std() if null.std() > 0 else np.nan
	print(f'  {label}')
	print(f'    edges tested      : {len(i)}')
	print(f'    observed mean JSD : {observed:.4f}')
	print(f'    null mean JSD     : {null.mean():.4f} (sd {null.std():.4f})')
	print(f'    z = {z:.2f}, one-sided p = {p_value:.4f}')
	return {'test': label, 'n_edges': len(i), 'observed_jsd': observed,
		'null_jsd': null.mean(), 'z': z, 'p_value': p_value}


def build_profiles(df, doc_topics, topic_cols):
	"""Topic profile per person, from every dissertation they're credited on.

	Built over BOTH roles, not advisors only: the network's nodes are defined by
	co-service, so the profiles have to cover the same population or the assortativity
	test loses nearly all its edges. The advisor-only view is written separately below.
	"""
	num_topics = len(topic_cols)
	merged = df.merge(doc_topics, on='record_id', how='inner')
	print(f'{len(merged)} modeled documents joined to advisor/committee metadata.')

	totals, counts, display, as_advisor = {}, {}, {}, {}
	for _, row in merged.iterrows():
		dist = row[topic_cols].to_numpy(dtype=float)
		for key, (name, roles) in people_on(row).items():
			display.setdefault(key, name)
			totals[key] = totals.get(key, np.zeros(num_topics)) + dist
			counts[key] = counts.get(key, 0) + 1
			if 'advisor' in roles:
				as_advisor[key] = as_advisor.get(key, 0) + 1

	rows = []
	for key, total in totals.items():
		if counts[key] < MIN_DISSERTATIONS:
			continue
		dist = total / total.sum()
		dominant = int(np.argmax(dist))
		row = {
			'key': key,
			'name': display[key],
			'n_dissertations': counts[key],
			'n_as_advisor': as_advisor.get(key, 0),
			'dominant_topic': dominant,
			'dominant_topic_share': float(dist[dominant]),
			'entropy': float(-(dist * np.log(dist + 1e-12)).sum()),
		}
		row.update({f'topic_{t}': dist[t] for t in range(num_topics)})
		rows.append(row)
	return pd.DataFrame(rows).sort_values('n_dissertations', ascending=False)


def run_assortativity(profiles, num_topics):
	G = nx.read_graphml(GRAPH_PATH)
	idx = {key: n for n, key in enumerate(profiles['key'])}
	P = profiles[[f'topic_{t}' for t in range(num_topics)]].to_numpy(dtype=float)
	program = dict(zip(profiles['key'], profiles['key'].map(
		lambda k: G.nodes[k]['top_program'] if k in G.nodes else '')))

	edges = [(u, v) for u, v in G.edges() if u in idx and v in idx]
	if len(edges) < 30:
		print(f'Only {len(edges)} edges have topic profiles at both ends; skipping test.')
		return pd.DataFrame()

	print(f'\n=== TOPIC ASSORTATIVITY (do co-servers share topics?) ===')
	print(f'Graph edges: {G.number_of_edges()}, usable (both endpoints profiled): {len(edges)}')

	results = []
	i = np.array([idx[u] for u, v in edges])
	j = np.array([idx[v] for u, v in edges])
	results.append(permutation_test(P, i, j, 'all co-service edges'))

	# Advisor-backed ties only: excludes pairs who merely sat on the same committee.
	strong = [(u, v) for u, v in edges if G[u][v].get('strongest_kind') != 'cc']
	if len(strong) >= 30:
		results.append(permutation_test(
			P, np.array([idx[u] for u, v in strong]), np.array([idx[v] for u, v in strong]),
			'advisor-backed edges only (aa/ac)'))

	# The department control. Topic models largely recover discipline and advisors sit
	# in departments, so same-program edges would look assortative even if ties carried
	# no information. Restricting to CROSS-program edges asks whether co-service
	# predicts topical similarity beyond shared departmental membership.
	cross = [(u, v) for u, v in edges
		if program.get(u) and program.get(v) and program[u] != program[v]]
	if len(cross) >= 30:
		results.append(permutation_test(
			P, np.array([idx[u] for u, v in cross]), np.array([idx[v] for u, v in cross]),
			'cross-program edges only (department controlled)'))

	return pd.DataFrame(results)


def plot_heatmap(profiles, num_topics, top_n=30):
	advisors = profiles[profiles['n_as_advisor'] > 0].head(top_n)
	if advisors.empty:
		return
	matrix = advisors[[f'topic_{t}' for t in range(num_topics)]].to_numpy()

	plt.figure(figsize=(max(8, num_topics * 0.45), max(6, len(advisors) * 0.3)))
	plt.imshow(matrix, aspect='auto', cmap='viridis')
	plt.colorbar(label="Share of advisor's topic mass")
	plt.yticks(range(len(advisors)), advisors['name'], fontsize=7)
	plt.xticks(range(num_topics), range(num_topics))
	plt.xlabel('Topic')
	plt.title(f'Topic distribution for top {len(advisors)} advisors by dissertation count')
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / 'advisor_topic_heatmap.png', dpi=150)
	plt.close()


def plot_network_by_topic(profiles, num_topics, top_n=150):
	G = nx.read_graphml(GRAPH_PATH)
	dominant = dict(zip(profiles['key'], profiles['dominant_topic']))

	components = sorted(nx.connected_components(G), key=len, reverse=True)
	if not components:
		return
	giant = G.subgraph(components[0]).copy()
	top_nodes = [n for n, _ in sorted(giant.degree(), key=lambda kv: kv[1], reverse=True)[:top_n]]
	legible = giant.subgraph(top_nodes).copy()

	colored = [n for n in legible.nodes() if n in dominant]
	if not colored:
		print('No overlap between topic profiles and the plotted subgraph; skipping plot.')
		return

	pos = nx.spring_layout(legible, k=0.4, seed=2026)
	other = [n for n in legible.nodes() if n not in dominant]

	plt.figure(figsize=(14, 14))
	nx.draw_networkx_edges(legible, pos, alpha=0.15)
	nx.draw_networkx_nodes(legible, pos, nodelist=other, node_size=40,
		node_color='lightgray', alpha=0.6)
	nx.draw_networkx_nodes(legible, pos, nodelist=colored, node_size=70,
		node_color=[dominant[n] for n in colored], cmap=matplotlib.colormaps['tab20'],
		vmin=0, vmax=num_topics - 1, alpha=0.9)
	plt.title("Co-service network colored by dominant topic\n"
		f'(gray = fewer than {MIN_DISSERTATIONS} modeled dissertations)')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / 'advisor_network_by_topic.png', dpi=150)
	plt.close()
	print(f'Colored {len(colored)} of {legible.number_of_nodes()} plotted nodes by dominant topic.')


def main():
	df = pd.read_csv(DATA_PATH, low_memory=False)[['record_id', 'advisors', 'committee']]
	doc_topics = pd.read_csv(DOC_TOPICS_PATH)
	topic_cols = [c for c in doc_topics.columns if c.startswith('topic_')]
	num_topics = len(topic_cols)

	profiles = build_profiles(df, doc_topics, topic_cols)
	profiles.to_csv(OUTPUT_DIR / 'advisor_topics.csv', index=False)
	print(f'Wrote {len(profiles)} topic profiles '
		f'(people with >= {MIN_DISSERTATIONS} modeled dissertations).')

	# Entropy is confounded by n: someone with 5 dissertations can only spread topic
	# mass so far, so they look like a "specialist" for purely arithmetic reasons.
	corr = profiles['entropy'].corr(np.log(profiles['n_dissertations']))
	print(f'Correlation(entropy, log n_dissertations) = {corr:.3f} '
		f'-- if strongly positive, do NOT read entropy as intellectual breadth.')

	results = run_assortativity(profiles, num_topics)
	if not results.empty:
		results.to_csv(OUTPUT_DIR / 'assortativity_results.csv', index=False)

	plot_heatmap(profiles, num_topics)
	plot_network_by_topic(profiles, num_topics)


if __name__ == '__main__':
	main()
