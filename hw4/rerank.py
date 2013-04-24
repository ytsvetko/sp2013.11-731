#!/usr/bin/env python
import optparse
import sys
import random
import json
import bleu
import collections
import os

optparser = optparse.OptionParser()

optparser.add_option("-p", "--creg_predictions", dest="creg_predictions_file",
    default="work/test.predicted", help="output of creg")

optparser.add_option("-t", "--test_hyp_file", dest="test_hyp_file",
    default="data/test.100best", help="Hypos for test")

optparser.add_option("-o", "--output_file", dest="out_file",
    default="output.txt", help="Output file with best hypo per sentence")

(opts, _) = optparser.parse_args()

class Hypothesis:
  def __init__(self, line):
    self.sent_id, text, features = line.strip().split(' ||| ')
    self.words = text.split()
    self.features = {}
    for feat in features.split(' '):
      k,v = feat.split('=')
      self.features[k] = float(v)

class Sentence:  # collection of 100 hypotheses.
  def __init__(self, hypos):
    self.hypos = hypos
    self.sent_id = hypos[0].sent_id
    for hyp in hypos:
      assert (self.sent_id == hyp.sent_id), (self.sent_id, hyp.sent_id)

def ReadSentences(hyp_filename):
  sentences = []
  all_hyps = [Hypothesis(line) for line in open(hyp_filename)]
  num_sents = len(all_hyps)/100
  for s in xrange(0, num_sents):
    sentences.append(Sentence(all_hyps[s * 100:s * 100 + 100]))
  return sentences

def ReadCregPredictions(filename):
  hypos_scores = []
  for line in open(filename):
    line_tokens = line.split("\t")
    if len(line_tokens) == 2:
      # Creg output in a linear regression mode
      hypos_scores.append(float(line_tokens[1]))
    else:
      # Creg output in logistic regression mode
      hypos_scores.append(json.loads(line_tokens[2])['1'])
  result = []
  num_sents = len(hypos_scores) / 100
  for i in xrange(num_sents):
    sent_hypos = hypos_scores[i*100:i*100+100]
    best, best_score = max(enumerate(sent_hypos), key=lambda x: x[1])
    result.append(best)
  return result
    
def main():
  print "Loading creg predictions"
  best_hypos = ReadCregPredictions(opts.creg_predictions_file)
  print "Outputing best hypos"
  out_file = open(opts.out_file, "w")
  for index, sentence in enumerate(ReadSentences(opts.test_hyp_file)):
    hyp = sentence.hypos[best_hypos[index]]
    out_file.write("{}\n".format(" ".join(hyp.words)))

if __name__ == '__main__':
    main()
