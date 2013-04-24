#!/usr/bin/env python
import optparse
import sys
import random
import json
import bleu

optparser = optparse.OptionParser()

optparser.add_option("-i", "--input_hypos", dest="input",
    default="data/dev.100best", help="100-best translation lists")

optparser.add_option("-r", "--input_ref", dest="ref_file",
    default="data/dev.ref", help="reference sentences")

optparser.add_option("-f", "--input_src", dest="src_file",
    default="data/dev.src", help="Russian sentences")

optparser.add_option("-o", "--output_training_samples_file", 
    dest="output_training_samples_file", default=None,
    help="output file with training samples (sent_id, first, second, distance)")

optparser.add_option("-n", "--num_training_samples",
    dest="num_training_samples", default=400, type="int",
    help="Number of traning instances per sentence.")

optparser.add_option("-c", "--num_pairs_to_consider",
    dest="num_pairs_to_consider", default=10000, type="int",
    help="Number of trials to select a random pair per sentence.")

optparser.add_option("--seed", dest="seed", default=123456, type="int",
    help="Random generator initializer")

(opts, _) = optparser.parse_args()

class BlueScorer:
  def Score(self, hyp, ref):
    stats = [0 for i in xrange(10)]
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(hyp,ref))]
    return bleu.bleu(stats)

DefScorer = BlueScorer()

class Hypothesis:
  def __init__(self, line):
    self.sent_id, text, features = line.strip().split(' ||| ')
    self.words = text.split()
    self.features = {}
    for feat in features.split(' '):
      k,v = feat.split('=')
      self.features[k] = float(v)

class Sentence:  # collection of 100 hypotheses.
  def __init__(self, hypos, ref):
    self.hypos = hypos
    self.ref = ref.split()
    self.sent_id = hypos[0].sent_id
    for hyp in hypos:
      assert (self.sent_id == hyp.sent_id), (self.sent_id, hyp.sent_id)

  def RandomSamples(self, num_returned_samples, num_considered_samples=opts.num_pairs_to_consider):
    seen_scores = {}
    def GetScore(hypo_num):
      if hypo_num in seen_scores:
        return seen_scores[hypo_num]
      seen_scores[hypo_num] = DefScorer.Score(self.hypos[hypo_num].words, self.ref)
      return seen_scores[hypo_num]
    seen_samples = {}   # key=first,second, value=distance
    for _ in xrange(num_considered_samples):
      first = random.randint(0, len(self.hypos)-1)
      second = random.randint(0, len(self.hypos)-1)
      if first == second:
        continue
      if (first, second) in seen_samples or (second, first) in seen_samples:
        continue
      first_score = GetScore(first)
      second_score = GetScore(second)
      score_distance = first_score - second_score
      if score_distance == 0.0 or first_score == 0.0 or second_score == 0.0:
        continue
      seen_samples[(first, second)] = score_distance
    sorted_samples = sorted(seen_samples.iteritems(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_samples[:num_returned_samples]

def ReadSentences(hyp_filename, ref_filename):
  sentences = []
  all_hyps = [Hypothesis(line) for line in open(hyp_filename)]
  all_refs = [line.strip() for line in open(ref_filename)]
  num_sents = len(all_refs)
  assert num_sents*100 == len(all_hyps)
  for s in xrange(0, num_sents):
    sentences.append(Sentence(all_hyps[s * 100:s * 100 + 100], all_refs[s]))
  return sentences

def main():
  random.seed(opts.seed)
  if opts.output_training_samples_file is None:
    output_training_file = sys.stdout
  else:
    output_training_file = open(opts.output_training_samples_file, "w")
  output_training_file.write("{}\t{}\t{}\n".format(opts.input, opts.ref_file, opts.src_file))
  for index, sentence in enumerate(ReadSentences(opts.input, opts.ref_file)):
    sys.stderr.write(".")
    for (first, second), distance in sentence.RandomSamples(opts.num_training_samples):
      output_training_file.write(
          "{sent_id}\t{first}\t{second}\t{distance}\n".format(
              sent_id=sentence.sent_id,
              first=first,
              second=second,
              distance=distance))
  sys.stderr.write("\n")

if __name__ == '__main__':
    main()
