#!/usr/bin/env python
import optparse
import sys
import codecs
import os
import collections

optparser = optparse.OptionParser()

optparser.add_option("-i", "--input_files", dest="input_files",
    default="data/brown_test.100best,data/brown_dev.100best",
    help="Hypos for test")

optparser.add_option("-o", "--output_file_prefix", dest="output_file_prefix",
    default="work/bigram_", help="Prefix for output files with lm features")

optparser.add_option("-f", "--feature_name_prefix", dest="feature_name_prefix",
    default="brown_bigram_", help="Feature name")

(opts, _) = optparser.parse_args()

def CalcBigramFeatures(words):
  result = {}
  for i in xrange(len(words)-1):
    bigram = "_".join(words[i:i+2])
    result[opts.feature_name_prefix + bigram] = 1
  return result

def ProcessFile(in_filename, out_filename):
  out_file = open(out_filename, "w")
  for line in codecs.open(in_filename):
    line = line.strip().split(" ||| ")
    if len(line) > 1:
      line = line[1]
    else:
      line = line[0]
    features = CalcBigramFeatures(line.split())
    str_features = ["{}\t{}".format(k,v) for (k,v) in features.iteritems()]
    out_file.write("\t".join(str_features) + "\n")

def main():
  print "Processing input files"
  for filename in opts.input_files.split(","):
    print filename
    out_filename = opts.output_file_prefix + os.path.basename(filename)
    ProcessFile(filename, out_filename)

if __name__ == '__main__':
    main()
