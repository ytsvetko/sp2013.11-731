#!/bin/bash

CREG=/home/ytsvetko/tools/creg/dist/bin/creg
export python=/usr/bin/pypy
WORKDIR=./work
mkdir -p $WORKDIR

RANDOM_SEED=123456

# These numbers are per sentence
NUM_PAIRS_TO_CONSIDER=10000
NUM_TOP_PAIRS_TO_OUTPUT=100

EXTRACT_LM_FEATURES=0
USE_BLEU_SCORE=0

CREG_PARAMS="-D "
CREG_PARAMS+="--l2 5.0 "
#CREG_PARAMS+="-n "   # Linear regression

# Extract LM features (slooooow)
if [ "$EXTRACT_LM_FEATURES" == "1" ]; then
  echo "English Brown clusters LM"
  $python ./lm_features.py -l data/brown/mt/lm/brown_en.4grams.arpa \
     -i data/brown_test.100best,data/brown_dev.100best -o work/lm_ -f brown_lm
  echo "English LM"
  $python ./lm_features.py -l /home/ytsvetko/workhorse4/projects/wmt13/data/input/lm/wmt13_all.4grams/wmt13_all.4grams.arpa \
     -i data/test.100best,data/dev.100best -o work/lm_ -f big_lm

  # The following two are not needed.
  echo "Russian Brown cluster LM"
  $python ./lm_features.py -l data/brown/mt/lm/brown_ru.4grams.arpa \
     -i data/brown_test.src,data/brown_dev.src -o work/lm_ -f ru_brown_lm
  echo "Russian LM"
  $python ./lm_features.py -l /home/ytsvetko/workhorse4/projects/wmt13/data/input/lm/wmt13_all_ru.4grams/wmt13_all_ru.4grams.arpa \
     -i data/test.src,data/dev.src -o work/lm_ -f ru_big_lm
fi

# Collect samples
if [ "$USE_BLEU_SCORE" == "1" ]; then
  $python ./select_samples_bleu.py -i data/dev.100best -r data/dev.ref --seed $RANDOM_SEED \
    -f data/dev.src -n $NUM_TOP_PAIRS_TO_OUTPUT -c $NUM_PAIRS_TO_CONSIDER -o $WORKDIR/samples.txt
else
  $python ./select_samples_meteor.py -i data/dev.100best -r data/dev.ref --seed $RANDOM_SEED \
    -f data/dev.src -n $NUM_TOP_PAIRS_TO_OUTPUT -c $NUM_PAIRS_TO_CONSIDER -o $WORKDIR/samples.txt
fi

# Generate Creg files
$python ./pro_samples.py -t data/test.100best -s $WORKDIR/samples.txt -o $WORKDIR

# Run Creg
$CREG -x $WORKDIR/train.feat -y $WORKDIR/train.label $CREG_PARAMS \
    --z $WORKDIR/train.weights --tx $WORKDIR/test.feat > $WORKDIR/test.predicted

b=""
# Select best hypos
$python ./rerank.py -t data/test.100best -p $WORKDIR/test.predicted -o output.txt

# Score
echo "Bleu:"
$python ./score-bleu < output.txt
./score-meteor < output.txt | tail -19

