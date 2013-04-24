#!/usr/bin/env python
import optparse
import sys
import codecs
import os
import models   # From HW3

optparser = optparse.OptionParser()

optparser.add_option("-l", "--lm_file", dest="lm_file",
    default="data/brown/mt/lm/brown_en.4grams.arpa",
    help="Name of the LM")

optparser.add_option("-i", "--input_files", dest="input_files",
    default="data/brown_test.100best,data/brown_test.ref,data/brown_dev.ref,data/brown_dev.100best",
    help="Hypos for test")

optparser.add_option("-o", "--output_file_prefix", dest="output_file_prefix",
    default="work/lm_", help="Prefix for output files with lm features")

optparser.add_option("-f", "--feature_name", dest="feature_name",
    default="brown_lm", help="Feature name")

(opts, _) = optparser.parse_args()

def CalcLmScore(words, lm):   # Code below is copied from HW3.
  lm_state = lm.begin() # initial state is always <s>
  logprob = 0.0
  for word in words:
    (lm_state, word_logprob) = lm.score(lm_state, word)
    logprob += word_logprob
  logprob += lm.end(lm_state) # transition to </s>
  return logprob

def ProcessFile(in_filename, out_filename, lm):
  out_file = open(out_filename, "w")
  for line in codecs.open(in_filename):
    line = line.strip().split(" ||| ")
    if len(line) > 1:
      line = line[1]
    else:
      line = line[0]
    lm_score = CalcLmScore(line.split(), lm)
    out_file.write("{}\t{}\n".format(opts.feature_name, lm_score))

def main():
  print "Loading LM"
  lm = models.LM(opts.lm_file)
  print "Processing input files"
  for filename in opts.input_files.split(","):
    print filename
    out_filename = opts.output_file_prefix + os.path.basename(filename)
    ProcessFile(filename, out_filename, lm)

if __name__ == '__main__':
    main()
