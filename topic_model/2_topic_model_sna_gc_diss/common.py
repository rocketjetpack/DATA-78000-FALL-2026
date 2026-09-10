#!/usr/bin/env python3

import re

import pandas as pd


NAME_SPLIT_RE = re.compile(r';')

# Name normalization is deliberately shallow: casefold, drop periods/commas, collapse
# whitespace. It will NOT merge spelling/initial variants of the same person (e.g.
# "Wm. Battersby" vs "William Battersby") -- that needs fuzzy matching, which risks
# incorrectly merging distinct people who share a name, so it's left as a stated
# limitation rather than guessed at silently.


def normalize_name(name):
	name = re.sub(r'[.,]', '', name)
	name = re.sub(r'\s+', ' ', name).strip()
	return name.casefold()


def split_people(field):
	if pd.isna(field) or not isinstance(field, str):
		return []
	return [p.strip() for p in NAME_SPLIT_RE.split(field) if p.strip()]


def people_on(row):
	"""Everyone credited on one dissertation, as {key: (display_name, roles)}.

	A person listed as both advisor and committee member on the same record gets
	both roles but is counted once, so they never form a self-edge.
	"""
	found = {}
	for column, role in (('advisors', 'advisor'), ('committee', 'committee')):
		for raw in split_people(row.get(column)):
			key = normalize_name(raw)
			if not key:
				continue
			if key not in found:
				found[key] = (raw, set())
			found[key][1].add(role)
	return found
