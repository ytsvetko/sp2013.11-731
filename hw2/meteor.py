#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import collections
 
def count_chunks(hyp, ref):
  chunks = 0
  mapped_unigrams = 0

  hyp_dict = collections.defaultdict(list)
  for ind, word in enumerate(hyp):
    hyp_dict[word].append(ind)

  prev_hyp_ind = -2
  curr_chunk = []
  for r in ref:
    if  r in hyp_dict:
      if len(curr_chunk) > 0:
        if prev_hyp_ind+1 in hyp_dict[r]:
          curr_chunk.append(r)
          prev_hyp_ind += 1
        else:
          chunks += 1
          mapped_unigrams += len(curr_chunk)
          curr_chunk = [r]
          prev_hyp_ind = hyp_dict[r][0]
      else:
        curr_chunk = [r]
        prev_hyp_ind = hyp_dict[r][0]
    elif len(curr_chunk)>0 :
      chunks += 1
      mapped_unigrams += len(curr_chunk)
      prev_hyp_ind = -2
      curr_chunk = []    
  return  chunks, mapped_unigrams     
   
def evaluate(hyp, ref):
  chunks, mapped_unigrams = count_chunks(hyp, ref)
  if mapped_unigrams == 0: 
    p = 1
  else: 
    p = 0.6 * (1.0*chunks/mapped_unigrams)**0.2 # weights from multieval
  return f_mean(hyp, ref) * (1 - p)

def f_mean(hyp, ref):
  h_set = set(hyp)
  r_set = set(ref)
  match = h_set.intersection(r_set)
  precision =  len(match)/float(len(hyp)) 
  recall =  len(match)/float(len(ref))
  if precision+recall == 0 :
    return 0
  return precision*recall/(0.85*precision+0.25*recall)# weights from multieval

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
