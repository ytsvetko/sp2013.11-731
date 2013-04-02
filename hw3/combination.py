#!/usr/bin/env python
import sys
import glob
import collections

outputs = glob.glob("*.scores")
results = collections.defaultdict(list)

for output in outputs:
  for line_num, line in enumerate(open(output)):
    tok = line.strip().split("\t")
    if len(tok) != 2:
      continue 
    results[line_num].append((float(tok[0]), tok[1]))

for line_num in range(55):
  hypos = sorted(results[line_num], key=lambda tup: tup[0], reverse=True) 
  sys.stdout.write("%s\n" % (hypos[0][1]))

