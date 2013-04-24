#!/usr/bin/env python
import optparse
import sys
import random
import json
import bleu
import codecs
import collections
import os
import math
from itertools import izip

optparser = optparse.OptionParser()

optparser.add_option("-s", "--training_samples_file", dest="training_samples_file",
    default="samples.txt", help="output of the select_samples.py")

optparser.add_option("-t", "--test_hyp_file", dest="test_hyp_file",
    default="data/test.100best", help="Hypos for test")

optparser.add_option("-f", "--test_src_file", dest="test_src_file",
    default="data/test.src", help="Russian test file")

optparser.add_option("-o", "--output_dir", dest="out_dir",
    default="./work", help="Output directory for creg files")

(opts, _) = optparser.parse_args()

PUNCT=".'/\"?!.:;-()[]%0123456789"

class Hypothesis:
  def __init__(self, line):
    self.sent_id, self.hyp_text, features = line.strip().split(' ||| ')
    self.words = self.hyp_text.split()
    self.features = {}
    for feat in features.split(' '):
      k,v = feat.split('=')
      self.features[k] = float(v)  # math.exp(-float(v))

  def CalculateFeatures(self, src):
    src_text = " ".join(src)

    def HasCyrilics(word):
      for ch in word:
        if ord(ch) > 255:
          return True
      return False

    def PunctFeatures():
      result = {}
      for p in PUNCT:
        num_in_eng = self.hyp_text.count(p) + 1
        num_in_src = src_text.count(p) + 1
        result["punct"+p] = 1.0 * num_in_src / num_in_eng
      return result

    self.features["len_src/trgt"] = len(src)*1.0/len(self.words)
    self.features["len_trgt/src"] = len(self.words)/len(src)*1.0
    self.features["len_trgt"] = len(self.words)
    self.features["num_untranslated"] = len(filter(HasCyrilics, self.words))
    self.features.update(PunctFeatures())
    """
    for feat, val in self.features.items():
      if val > 0.0:
        self.features["log_"+feat] = math.log(val)
    """

class TrainingSample:
  def __init__(self, first_hyp, second_hyp, bool_label):
    self.first_hyp = first_hyp
    self.second_hyp = second_hyp
    self.label = 1 if bool_label else -1

  def GetFeatures(self):
    feat_dict = {}
    all_feat = set(self.first_hyp.features.iterkeys())
    all_feat.update(self.second_hyp.features.iterkeys())

    for feat in all_feat:
      first_val = self.first_hyp.features.get(feat, 0.0)
      second_val = self.second_hyp.features.get(feat, 0.0)
      feat_dict[feat] = (first_val - second_val)
    return feat_dict

class Sentence:  # collection of 100 hypotheses.
  def __init__(self, hypos, src):
    self.hypos = hypos
    self.src = src
    self.sent_id = hypos[0].sent_id
    for hyp in hypos:
      assert (self.sent_id == hyp.sent_id), (self.sent_id, hyp.sent_id)

  def UpdateFeatures(self):
    for hyp in self.hypos:
      hyp.CalculateFeatures(self.src)

def LoadExtraFeatures(hyp_files):
  result = {}
  for f in hyp_files:
    tokens = f.readline().strip().split("\t")
    k_v_pairs = izip(tokens[::2], tokens[1::2])
    k_v_pairs = {(k, math.exp(float(v))) for (k,v) in k_v_pairs}
    result.update(k_v_pairs)
  return result

def ReadSentences(hyp_filename, src_filename, hyp_feature_filenames):
  hyp_feature_files = [open(filename) for filename in hyp_feature_filenames]
  sentences = []
  all_hyps = []
  for line in codecs.open(hyp_filename, "r", "utf-8"):
    hyp = Hypothesis(line)
    hyp.features.update(LoadExtraFeatures(hyp_feature_files))
    all_hyps.append(hyp)
  all_srcs = [line.strip().split() for line in codecs.open(src_filename, "r", "utf-8")]
  num_sents = len(all_hyps)/100
  assert num_sents == len(all_srcs)
  for s in xrange(0, num_sents):
    sent = Sentence(all_hyps[s * 100:s * 100 + 100], all_srcs[s])
    sent.UpdateFeatures()
    sentences.append(sent)
  return sentences

def WriteCregFiles(data, labels, fileprefix):
  feat_file = open(fileprefix + ".feat", "w")
  if labels is not None:
    label_file = open(fileprefix + ".label", "w")
  else:
    labels = xrange(len(data))
    label_file = None
  for index, (feat, label) in enumerate(zip(data, labels)):
    feat_file.write("{}\t{}\n".format(index, json.dumps(feat)))
    if label_file is not None:
      label_file.write("{}\t{}\n".format(index, label))

def ReadSamples(filename):
  f = open(filename)
  hyp_file, ref_file, src_file = f.readline().strip().split("\t")
  result = collections.defaultdict(list) # key=sent_id, value=list_of(first, second, distance)
  for line in f:
    sent_id, first, second, distance = line.strip().split("\t")
    result[sent_id].append( (int(first), int(second), float(distance)) )
  return hyp_file, ref_file, src_file, result

def ProcessFiles(hyp_file, src_file, creg_file_prefix, samples):
  data = []
  labels = []
  for index, sentence in enumerate(ReadSentences(hyp_file, src_file,
      ["work/lm_brown_dev.100best", "work/meteor_dev.100best"])):
    for first, second, distance in samples[sentence.sent_id]:
      training_instance = TrainingSample(sentence.hypos[first], sentence.hypos[second], distance > 0.0)
      data.append(training_instance.GetFeatures())
      labels.append(training_instance.label)
      training_instance = TrainingSample(sentence.hypos[second], sentence.hypos[first], distance < 0.0)
      data.append(training_instance.GetFeatures())
      labels.append(training_instance.label)
    del samples[sentence.sent_id]
  assert len(samples) == 0, samples
  WriteCregFiles(data, labels, creg_file_prefix)

def main():
  print "Loading samples"
  hyp_file, ref_file, src_file, samples = ReadSamples(opts.training_samples_file)
  print "Train"
  ProcessFiles(hyp_file, src_file, os.path.join(opts.out_dir, "train"), samples)
  print "Test"
  data = []
  for index, sentence in enumerate(ReadSentences(opts.test_hyp_file, opts.test_src_file, 
      ["work/lm_brown_test.100best", "work/meteor_test.100best"])):
    for hyp in sentence.hypos:
      data.append(hyp.features)
  WriteCregFiles(data, None, os.path.join(opts.out_dir, "test"))

if __name__ == '__main__':
    main()
