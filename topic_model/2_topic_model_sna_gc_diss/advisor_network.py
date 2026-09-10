#!/usr/bin/env python3

import sys
from pathlib import Path
from itertools import combinations
from collections import Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from common import people_on


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'gc_dissertations_combined_v2.csv'
OUTPUT_DIR = SCRIPT_DIR / 'output'

print('Data source: ', DATA_PATH)

if not DATA_PATH.exists():
	print('Data source is not available or does not exist.')
	sys.exit(1)

OUTPUT_DIR.mkdir(exist_ok=True)

# Edge definition: two people are connected if they served on the same dissertation,
# as advisor(s) or committee members. Each edge records which kind of co-service
# produced it, because the three kinds are NOT equivalent evidence:
#   aa (advisor-advisor)     - both formally advised the student; strongest tie
#   ac (advisor-committee)   - one advised, the other sat on the committee
#   cc (committee-committee) - both merely sat on the same committee; weakest tie,
#                              and neither necessarily chose the other
EDGE_KINDS = ('aa', 'ac', 'cc')

# Era scoping. The `committee` column is absent from every proquest_legacy record and
# present in 59.3% of academic_works records, which makes committee coverage a function
# of cataloging regime, not of advising practice: 1.6-4.4% of records per decade before
# 2010, then 35.2% (2010s) and 61.6% (2020s). Since ~95% of edges depend on committee
# data, the co-service network is effectively a post-2010 object. We therefore build it
# per era and make that limitation visible rather than implying 60 years of coverage.
ERAS = {
	'all': (0, 9999),
	'pre2010': (0, 2009),
	'post2010': (2010, 9999),
}
PRIMARY_ERA = 'post2010'

TOP_N_FOR_PLOT = 150  # a fixed degree threshold doesn't thin this graph -- even
# degree>=3 leaves 5,254 of 5,835 giant-component nodes -- so cap the node count directly


def build_graph(df):
	G = nx.Graph()

	for _, row in df.iterrows():
		found = people_on(row)
		if not found:
			continue

		for key, (display, roles) in found.items():
			if not G.has_node(key):
				G.add_node(key, name=display, dissertations=0, n_as_advisor=0,
					n_as_committee=0, programs=Counter(), years=[])
			node = G.nodes[key]
			node['dissertations'] += 1
			if 'advisor' in roles:
				node['n_as_advisor'] += 1
			if 'committee' in roles:
				node['n_as_committee'] += 1
			if pd.notna(row.get('program_coarse')):
				node['programs'][row['program_coarse']] += 1
			if pd.notna(row.get('year')):
				node['years'].append(row['year'])

		for a, b in combinations(sorted(found), 2):
			a_is_advisor = 'advisor' in found[a][1]
			b_is_advisor = 'advisor' in found[b][1]
			if a_is_advisor and b_is_advisor:
				kind = 'aa'
			elif a_is_advisor or b_is_advisor:
				kind = 'ac'
			else:
				kind = 'cc'

			if not G.has_edge(a, b):
				G.add_edge(a, b, weight=0, **{f'w_{k}': 0 for k in EDGE_KINDS})
			edge = G[a][b]
			edge['weight'] += 1
			edge[f'w_{kind}'] += 1

	for _, data in G.nodes(data=True):
		top_program = data['programs'].most_common(1)
		data['top_program'] = top_program[0][0] if top_program else ''
		data['n_programs'] = len(data['programs'])
		del data['programs']
		years = data.pop('years')
		data['year_min'] = int(min(years)) if years else 0
		data['year_max'] = int(max(years)) if years else 0

	# strongest_kind lets downstream analysis filter to advisor-backed ties only.
	for _, _, edge in G.edges(data=True):
		edge['strongest_kind'] = 'aa' if edge['w_aa'] else ('ac' if edge['w_ac'] else 'cc')

	return G


def graph_stats(G, label, n_rows):
	comps = sorted(nx.connected_components(G), key=len, reverse=True)
	kind_totals = Counter()
	for _, _, edge in G.edges(data=True):
		kind_totals[edge['strongest_kind']] += 1
	n_edges = max(G.number_of_edges(), 1)
	return {
		'era': label,
		'rows': n_rows,
		'nodes': G.number_of_nodes(),
		'edges': G.number_of_edges(),
		'isolated': sum(1 for _, d in G.degree() if d == 0),
		'giant': len(comps[0]) if comps else 0,
		'density': round(nx.density(G), 6),
		'edges_aa': kind_totals['aa'],
		'edges_ac': kind_totals['ac'],
		'edges_cc': kind_totals['cc'],
		'pct_advisor_backed': round(100 * kind_totals['aa'] / n_edges, 1),
	}


def summarize(G):
	print(f'Nodes (people): {G.number_of_nodes()}')
	print(f'Edges (co-service relations): {G.number_of_edges()}')
	print(f'Density: {nx.density(G):.6f}')

	components = sorted(nx.connected_components(G), key=len, reverse=True)
	print(f'Connected components: {len(components)}')
	print(f'Largest component size: {len(components[0])}')

	degrees = dict(G.degree())
	top_by_degree = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:15]
	print('Top 15 by degree (collaborator count):')
	for key, degree in top_by_degree:
		d = G.nodes[key]
		print(f"  {d['name']:<32} degree={degree:<4} diss={d['dissertations']:<4} "
			f"as_advisor={d['n_as_advisor']:<4} program={d['top_program']}")

	giant = G.subgraph(components[0])
	communities = list(nx.community.greedy_modularity_communities(giant, weight='weight'))
	print(f'Communities detected in largest component: {len(communities)}')
	print('Largest 5 community sizes:', sorted((len(c) for c in communities), reverse=True)[:5])

	# How much does community structure just restate the program column we already have?
	prog = {n: giant.nodes[n]['top_program'] for n in giant if giant.nodes[n]['top_program']}
	if prog:
		from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
		node2comm = {n: i for i, c in enumerate(communities) for n in c}
		common = [n for n in giant if n in prog]
		labels_p = [prog[n] for n in common]
		labels_c = [node2comm[n] for n in common]
		print(f'Community vs program agreement: '
			f'NMI={normalized_mutual_info_score(labels_p, labels_c):.3f} '
			f'ARI={adjusted_rand_score(labels_p, labels_c):.3f} '
			f'(1.0 would mean communities merely relabel departments)')
		print('Least program-pure communities (where co-service crosses departments):')
		purity = []
		for c in communities:
			progs = [prog[n] for n in c if n in prog]
			if len(progs) < 30:
				continue
			s = pd.Series(progs).value_counts()
			purity.append((100 * s.iloc[0] / len(progs), len(c), s.index[0],
				s.index[1] if len(s) > 1 else '-'))
		for pct, size, modal, second in sorted(purity)[:5]:
			print(f'  size={size:<5} purity={pct:5.1f}%  {modal} + {second}')

	return components, communities


def plot_network(G, components, era):
	giant = G.subgraph(components[0]).copy()
	top_nodes = [n for n, _ in sorted(giant.degree(), key=lambda kv: kv[1], reverse=True)[:TOP_N_FOR_PLOT]]
	legible = giant.subgraph(top_nodes)
	print(f'Plotting top {legible.number_of_nodes()} of {giant.number_of_nodes()} '
		f'giant-component nodes by degree for legibility.')

	pos = nx.spring_layout(legible, k=0.6, seed=2026)
	degrees = dict(legible.degree())
	sizes = [80 + 15 * degrees[n] for n in legible.nodes()]
	label_cutoff = sorted(degrees.values(), reverse=True)[min(39, len(degrees) - 1)]

	# Advisor-backed ties drawn solid, committee-only ties faint: the two are very
	# different evidence and shouldn't look identical.
	strong = [(u, v) for u, v, d in legible.edges(data=True) if d['strongest_kind'] != 'cc']
	weak = [(u, v) for u, v, d in legible.edges(data=True) if d['strongest_kind'] == 'cc']

	plt.figure(figsize=(14, 14))
	nx.draw_networkx_edges(legible, pos, edgelist=weak, alpha=0.12, edge_color='gray')
	nx.draw_networkx_edges(legible, pos, edgelist=strong, alpha=0.55, edge_color='darkred')
	nx.draw_networkx_nodes(legible, pos, node_size=sizes, node_color='steelblue', alpha=0.8)
	labels = {n: legible.nodes[n]['name'] for n in legible.nodes() if degrees[n] >= label_cutoff}
	nx.draw_networkx_labels(legible, pos, labels=labels, font_size=8)
	plt.title(f'Advisor/committee co-service network, era={era} '
		f'(top {TOP_N_FOR_PLOT} by degree)\n'
		f'red = advisor-backed tie, gray = committee-only tie')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR / 'advisor_network.png', dpi=150)
	plt.close()


def main():
	df = pd.read_csv(DATA_PATH, low_memory=False)

	print('\n=== ERA COMPARISON ===')
	stats = []
	for era, (lo, hi) in ERAS.items():
		sub = df[df['year'].between(lo, hi)]
		stats.append(graph_stats(build_graph(sub), era, len(sub)))
	stats_df = pd.DataFrame(stats)
	print(stats_df.to_string(index=False))
	stats_df.to_csv(OUTPUT_DIR / 'era_comparison.csv', index=False)

	lo, hi = ERAS[PRIMARY_ERA]
	print(f'\n=== PRIMARY GRAPH (era={PRIMARY_ERA}, years {lo}-{hi}) ===')
	df_era = df[df['year'].between(lo, hi)]
	G = build_graph(df_era)
	components, _ = summarize(G)
	plot_network(G, components, PRIMARY_ERA)

	nx.write_graphml(G, OUTPUT_DIR / 'advisor_network.graphml')

	component_id = {n: i for i, comp in enumerate(components) for n in comp}
	nodes_df = pd.DataFrame([
		{
			'key': key,
			'name': data['name'],
			'degree': G.degree(key),
			'dissertations': data['dissertations'],
			'n_as_advisor': data['n_as_advisor'],
			'n_as_committee': data['n_as_committee'],
			'top_program': data['top_program'],
			'year_min': data['year_min'],
			'year_max': data['year_max'],
			'component_id': component_id[key],
		}
		for key, data in G.nodes(data=True)
	])
	nodes_df.to_csv(OUTPUT_DIR / 'advisor_nodes.csv', index=False)
	print(f'Wrote {len(nodes_df)} node rows to {OUTPUT_DIR / "advisor_nodes.csv"}')


if __name__ == '__main__':
	main()
