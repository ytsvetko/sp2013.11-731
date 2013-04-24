#!/usr/bin/env python
import optparse
import sys
import os
import bleu
import subprocess

optparser = optparse.OptionParser()

optparser.add_option("-i", "--input_hypos", dest="input",
    default="data/dev.100best", help="100-best translation lists")

optparser.add_option("-r", "--input_ref", dest="ref_file",
    default="../wmt13/data/work/systems/baseline0-0/test/dev.src.out",
    help="reference sentences")

optparser.add_option("-o", "--output_feature_file",
    dest="output_feature_file", default="work/meteor_dev.100best",
    help="output file with features")

optparser.add_option("--temp_hyp_for_meteor", dest="temp_meteor_hyp",
   default="/tmp/temp_hyp", help="File with hypos for meteor")

optparser.add_option("--temp_ref_for_meteor", dest="temp_meteor_ref",
    default="/tmp/temp_ref", help="Ref file for meteor")

optparser.add_option("--meteor_output_prefix", dest="meteor_output_prefix",
    default="work/meteor_out_", help="Output of meteor")

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
  def __init__(self, hypos, ref):
    self.hypos = hypos
    self.ref = ref.split()
    self.sent_id = hypos[0].sent_id
    for hyp in hypos:
      assert (self.sent_id == hyp.sent_id), (self.sent_id, hyp.sent_id)

def ReadSentences(hyp_filename, ref_filename):
  sentences = []
  all_hyps = [Hypothesis(line) for line in open(hyp_filename)]
  all_refs = [line.strip() for line in open(ref_filename)]
  num_sents = len(all_refs)
  assert num_sents*100 == len(all_hyps)
  for s in xrange(0, num_sents):
    sentences.append(Sentence(all_hyps[s * 100:s * 100 + 100], all_refs[s]))
  return sentences

def WriteMeteorFiles(sentences, hyp_filename, ref_filename):
  hyp_file = open(hyp_filename, "w")
  ref_file = open(ref_filename, "w")
  for sentence in sentences:
    for hyp in sentence.hypos:
      ref_file.write(" ".join(sentence.ref) + "\n")
      hyp_file.write(" ".join(hyp.words) + "\n")
  hyp_file.close()
  ref_file.close()

def RunMeteor(sentences, hyp_filename, ref_filename, meteor_out_filename):
  meteor_jar = "./meteor-1.4/meteor-1.4.jar"
  meteor_command=["java", "-Xmx1G", "-jar " + meteor_jar, hyp_filename, ref_filename]
  sys.stderr.write("Running Meteor scorer\n")
  WriteMeteorFiles(sentences, hyp_filename, ref_filename)
  meteor_stdout = subprocess.check_output(" ".join(meteor_command), shell=True)
  open(meteor_out_filename, "w").write(meteor_stdout)
  return meteor_stdout.split("\n")
  
def LoadMeteorScores(sentences, hyp_filename, ref_filename, meteor_out_filename):
  try:
    meteor_stdout = open(meteor_out_filename).readlines()
    sys.stderr.write("Loaded Meteor scores from {}\n".format(meteor_out_filename))
  except IOError:
    meteor_stdout = RunMeteor(sentences, hyp_filename, ref_filename, meteor_out_filename)
  hyp_scores = []
  for line in meteor_stdout:
    if line.startswith("Segment "):
      hyp_scores.append(float(line.strip().split("\t")[-1]))
  sys.stderr.write("Found {} scores\n".format(len(hyp_scores)))
  return hyp_scores

def CalcBleuScores(sentences):
  def Score(hyp, ref):
    stats = [0 for i in xrange(10)]
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(hyp,ref))]
    return bleu.bleu(stats)
  result = []
  for sentence in sentences:
    for hyp in sentence.hypos:
      result.append(Score(sentence.ref, hyp.words))
  return result

def main():
  sentences = ReadSentences(opts.input, opts.ref_file)
  meteor_scores = LoadMeteorScores(sentences, opts.temp_meteor_hyp,
      opts.temp_meteor_ref,
      opts.meteor_output_prefix + os.path.basename(opts.input))
  bleu_scores = CalcBleuScores(sentences)
  out_file = open(opts.output_feature_file, "w")
  for m_score, b_score in zip(meteor_scores, bleu_scores):
    out_file.write("{}\t{}\t{}\t{}\n".format(
        "meteor", m_score, "bleu", b_score))

if __name__ == '__main__':
    main()
