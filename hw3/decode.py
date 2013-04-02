#!/usr/bin/env python
import argparse
import sys
import models
import heapq
from collections import namedtuple
import itertools

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-s', '--stack-size', dest='s', default=1, type=int, help='Maximum stack size (default=1)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,  help='Verbose mode (default=off)')
parser.add_argument('-d', '--reordering_distance_limit', dest='reordering_dist_limit', default=8, type=int, help='Distance limit on reordering of input words')
opts = parser.parse_args()

inf = float("inf")

def extract_english_recursive(h):
  return '' if h.predecessor is None else '%s%s ' % (extract_english_recursive(h.predecessor), h.phrase.english)

hypothesis = namedtuple('hypothesis', 'logprob, lm_state, predecessor, phrase')

class Bitmask:
  def __init__(self, bitlen, init_state, num_bits_in_state, rightmost_zero_location=None):
    self.bitlen = bitlen
    self.bitmask = init_state
    self.num_bits_in_state = num_bits_in_state
    self.rightmost_zero_location = rightmost_zero_location

  def Mark(self, index_from, index_to):
    assert 0 <= index_from < index_to < 64
    num_bits = index_to - index_from
    bits = (1 << num_bits) - 1
    rightmost_zero = self.rightmost_zero_location
    if index_from <= rightmost_zero <= index_to:
      rightmost_zero=None
    return Bitmask(self.bitlen, self.bitmask | bits << index_from,
                   self.num_bits_in_state + num_bits, rightmost_zero)

  def IsZeros(self, startpos, spanlen):
    right_aligned = self.bitmask >> startpos
    bits = (1 << spanlen) - 1
    return (right_aligned & bits) == 0

  def CalcRightmostZeroLocation(self):
    current_bit = 0
    bitmask = self.bitmask
    while bitmask & (1 << current_bit):
      current_bit += 1
    self.rightmost_zero_location = current_bit

  def GetZeroSpans(self):
    zero_start = None
    for i in xrange(self.bitlen):
      if self.bitmask & (1 << i):
        if zero_start is not None:
          yield (zero_start, i)
          zero_start = None
      elif zero_start is None:
        zero_start = i
    if zero_start is not None:
      yield (zero_start, self.bitlen)

  def iterspans(self):
    if self.rightmost_zero_location is None:
      self.CalcRightmostZeroLocation()
    for spanlen in xrange(self.bitlen, 0, -1):
      range_from = max(0, self.num_bits_in_state - opts.reordering_dist_limit)
      range_to = min(self.bitlen - spanlen+1,
                     opts.reordering_dist_limit + self.rightmost_zero_location)
      for startpos in xrange(range_from, range_to):
        if self.IsZeros(startpos, spanlen):
          yield (startpos, startpos + spanlen, self.num_bits_in_state + spanlen)

  def __hash__(self):
    return self.bitmask

  def __eq__(self, other):
    return self.bitmask == other.bitmask

  def __repr__(self):
    return bin(self.bitmask)[2:]

    
class LineDecoder:
  def __init__(self, tm, lm):
    self.tm = tm
    self.lm = lm
    self.initial_hypothesis = hypothesis(0.0, lm.begin(), None, None)

  def CalcCostEstimate(self, words):
    if words in self.tm:
      phrase = self.tm[words][0]
      return phrase.logprob, phrase.english.split()
    return inf, ""
    
  def CalcCostTable(self, f):
    cost = {}
    for l in xrange(1, len(f)+1):
      for start in xrange(len(f)-l+1):
        end = start+l
        tm_cost, en_phrase = self.CalcCostEstimate(f[start:end])
        for i in xrange(start+1, end):
          l_cost = cost[(start, i)]
          r_cost = cost[(i, end)]
          combined_cost = l_cost[0] + r_cost[0]
          if combined_cost < tm_cost:
            tm_cost = combined_cost
            en_phrase = l_cost[2] + r_cost[2]

        lm_cost = 0.0
        if len(en_phrase) > 0:
          lm_state = (en_phrase[0],)
          for word in en_phrase[1:]:
            (lm_state, word_logprob) = self.lm.score(lm_state, word)
            lm_cost += word_logprob

        cost[(start, end)] = (tm_cost, lm_cost, en_phrase)
    return cost

  def DecodeLine(self, f, stack_size, max_logprob):
    est_cost_table = self.CalcCostTable(f)

    def HypCost(kv):
      (bitmask, _), h = kv
      logprob = h.logprob
      for start, end in bitmask.GetZeroSpans():
        tm_cost, lm_cost, _ = est_cost_table[(start, end)]
        logprob += tm_cost #+ lm_cost
      if logprob < max_logprob:
        return -inf
      return logprob

    stacks = [{} for _ in f] + [{}]
    stacks[0][(Bitmask(len(f), 0, 0, 0), self.lm.begin())] = self.initial_hypothesis
    #print "Sentence length", len(f)
    for current_stack_num, stack in enumerate(stacks[:-1]):
      if current_stack_num > 0:
        stacks[current_stack_num-1] = {}
      #print "Processing stack", current_stack_num
      # extend the top s hypotheses in the current stack
      for (bitmask, lm_state), h in heapq.nlargest(stack_size, stack.iteritems(), key=HypCost): # prune
        for i,j,stack_num in bitmask.iterspans():
          #print i,j,stack_num 
          """
        i = current_stack_num
        for j in xrange(i+1,len(f)+1):
          stack_num = current_stack_num + (j-i)
        """
          if f[i:j] in self.tm:
            new_bitmask = bitmask.Mark(i,j)
            for phrase in self.tm[f[i:j]]:
              logprob = h.logprob + phrase.logprob
              lm_state = h.lm_state
              for word in phrase.english.split():
                (lm_state, word_logprob) = self.lm.score(lm_state, word)
                logprob += word_logprob
              logprob += self.lm.end(lm_state) if j == len(f) else 0.0                        
              key = (new_bitmask, lm_state)
              if logprob < max_logprob : 
                continue
              if key not in stacks[stack_num] or stacks[stack_num][key].logprob < logprob: # second case is recombination
                new_hypothesis = hypothesis(logprob, lm_state, h, phrase)
                stacks[stack_num][key] = new_hypothesis

    # find best translation by looking at the best scoring hypothesis
    # on the last stack
    if len(stacks[-1]) > 0:
      winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    else:
      return "", -inf
    return extract_english_recursive(winner), winner.logprob

def main():
  tm = models.TM(opts.tm, sys.maxint)
  lm = models.LM(opts.lm)
  sys.stderr.write('Decoding %s...\n' % (opts.input,))
  input_sents = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
  
  for sent_num, f in enumerate(input_sents):
    max_logprob = -inf
    curr_stack_size = 100
    curr_translation = ""
    while   curr_stack_size <= opts.s:
      sys.stderr.write("SentNum: {}, StackSize: {}\n".format(sent_num, curr_stack_size))
      translation, logprob = LineDecoder(tm, lm).DecodeLine(f, curr_stack_size, max_logprob)
      if logprob > max_logprob:
        curr_translation = translation
        max_logprob = logprob
      curr_stack_size = curr_stack_size*10
    print curr_translation

if __name__ == "__main__":
  main()
