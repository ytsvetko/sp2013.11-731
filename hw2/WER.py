#!/usr/bin/env python


import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import sys, itertools, math, collections

SUB='SUB'
LEFT='INS'
UP='DEL'
MATCH=''

MATCH_COST = 0

inf = float('inf')

def Levenshtein(ref, hyp):
  d, i, j = costMatrix(ref, hyp)

  # get the aligned strings
  hypAligned = []
  refAligned = []
  errList = []
  errNum = d[i][j][0]
  while (i>0 or j>0):
    assert i>=0
    assert j>=0
    if d[i][j][1]==UP: 
      i-=1
      hypAligned.insert(0, '')
      refAligned.insert(0, ref[i])
      errList.insert(0, UP)
    elif d[i][j][1]==LEFT: 
      j-=1
      hypAligned.insert(0, hyp[j])
      refAligned.insert(0, '')
      errList.insert(0, LEFT)
    elif d[i][j][1]==SUB: 
      i-=1
      j-=1
      hypAligned.insert(0, hyp[j])
      refAligned.insert(0, ref[i])
      errList.insert(0, SUB)
    else:
      i-=1
      j-=1
      hypAligned.insert(0, hyp[j])
      refAligned.insert(0, ref[i])
      errList.insert(0, MATCH)
  return refAligned, hypAligned, errList, errNum


def costMatrix(ref, hyp):
  d= [[(inf, MATCH) for j in xrange(1+len(hyp))] for i in xrange(1+len(ref))]
  d[0][0]=(0, MATCH) # each cell in matrix contains a cumulative path weight and a backpointer
  for i in xrange(1, len(ref)+1):
    d[i][0] = (d[i-1][0][0] + 1, UP)
  for j in xrange(1, len(hyp)+1):
    d[0][j] = (d[0][j-1][0] + 1, LEFT)
  for i in xrange(1, len(ref)+1):
    for j in xrange(1, len(hyp)+1):
      if hyp[j-1] == ref[i-1]:
        match_sub = (d[i-1][j-1][0], MATCH)
      else:
        match_sub = (d[i-1][j-1][0] + 1, SUB)
      candidates = [match_sub,
                    (d[i-1][j][0] + 1, UP),
                    (d[i][j-1][0] + 1, LEFT)]
      d[i][j] = min(candidates)
  return d, len(ref), len(hyp)


def evaluate(hyp, ref):
  # find best alignment using Levenshtein algorithm
  refAligned, hypAligned, errList, errNum =  Levenshtein(ref, hyp)
  return 1.0*errNum/len(ref) if len(ref) is not 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train.hyp1-hyp2-ref',
            help='input file (default data/train.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.lower().strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        h1_match = evaluate(h1, ref)
        h2_match = evaluate(h2, ref)
        print(-1 if h1_match > h2_match else # \begin{cases}
                (0 if h1_match == h2_match
                    else 1)) # \end{cases}
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
