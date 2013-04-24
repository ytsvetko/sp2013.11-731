#!/usr/bin/env python
import optparse
import sys
import random
import subprocess

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

optparser.add_option("--temp_hyp_for_meteor", dest="temp_meteor_hyp",
   default="/tmp/temp_hyp", help="File with hypos for meteor")

optparser.add_option("--temp_ref_for_meteor", dest="temp_meteor_ref",
    default="/tmp/temp_ref", help="Ref file for meteor")

optparser.add_option("--meteor_output", dest="meteor_output",
    default="work/meteor.out", help="Output of meteor")

optparser.add_option("--seed", dest="seed", default=123456, type="int",
    help="Random generator initializer")

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

  def RandomSamples(self, scores, num_returned_samples, num_considered_samples=opts.num_pairs_to_consider):
    seen_samples = {}   # key=first,second, value=distance
    for _ in xrange(num_considered_samples):
      first = random.randint(0, len(self.hypos)-1)
      second = random.randint(0, len(self.hypos)-1)
      if first == second:
        continue
      if (first, second) in seen_samples or (second, first) in seen_samples:
        continue
      first_score = scores[first]
      second_score = scores[second]
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
  result = []
  for i in xrange(0, len(hyp_scores), 100):
    result.append(hyp_scores[i:i+100])
  return result

def main():
  random.seed(opts.seed)
  sentences = ReadSentences(opts.input, opts.ref_file)
  scores = LoadMeteorScores(sentences, opts.temp_meteor_hyp,
      opts.temp_meteor_ref, opts.meteor_output)

  if opts.output_training_samples_file is None:
    output_training_file = sys.stdout
  else:
    output_training_file = open(opts.output_training_samples_file, "w")
  output_training_file.write("{}\t{}\t{}\n".format(opts.input, opts.ref_file, opts.src_file))

  for index, sentence in enumerate(sentences):
    sys.stderr.write(".")
    for (first, second), distance in sentence.RandomSamples(scores[index], opts.num_training_samples):
      output_training_file.write(
          "{sent_id}\t{first}\t{second}\t{distance}\n".format(
              sent_id=sentence.sent_id,
              first=first,
              second=second,
              distance=distance))
  sys.stderr.write("\n")

if __name__ == '__main__':
    main()
