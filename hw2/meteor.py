#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import collections
 
import  string


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
   
def evaluate(hyp, ref, synonyms):
  chunks, mapped_unigrams = count_chunks(hyp, ref)
  if mapped_unigrams == 0: 
    p = 1
  else: 
    p = 0.6 * (1.0*chunks/mapped_unigrams)**0.2 # weights from multieval
  return f_mean(hyp, ref, synonyms) * (1 - p)

def synonym_intersection(hyp, ref, synonyms):
  r_set = set(ref)
  count = 0
  for h in set(hyp): 
    syn = synonyms.get(h, [h])
    for s in syn:
      if s in r_set:
        count += 1
        break
  return count
 
def f_mean(hyp, ref, synonyms):
  if len(hyp) == 0 and len(ref) == 0:
    return 1
  if len(hyp) == 0 or len(ref) == 0:
    return 0
  match = synonym_intersection(hyp, ref, synonyms)
  precision =  match/float(len(hyp)) 
  recall =  match/float(len(ref))
  if precision+recall == 0 :
    return 0
  return precision*recall/(0.85*precision+0.25*recall)# weights from multieval

def remove_punctuation(s):
  return ''.join([i for i in s if i not in string.punctuation])

def preprocess(s) :
  return remove_punctuation(s.strip().lower())

def load_synonyms(filename):
  synonyms = collections.defaultdict(set)
  for line in open(filename):
    synset = line.split('\t')
    if len(synset)<2:
      continue
    for word in synset:
      synonyms[word].update(synset)
  return synonyms

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
                yield [preprocess(sentence).split() for sentence in pair.split(' ||| ')]
 
    synonyms = load_synonyms('data/wordnet_synonyms.en')
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        h1_match = evaluate(h1, ref, synonyms)
        h2_match = evaluate(h2, ref, synonyms)
        print(-1 if h1_match > h2_match else # \begin{cases}
                (0 if h1_match == h2_match
                    else 1)) # \end{cases}
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
