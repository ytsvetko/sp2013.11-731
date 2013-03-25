I implemented reordering, reordering limits, lookup tables with future cost estimation by TM score, by LM score. 

Experiments

- Initial implementation - -5697.455311

- Reordering, stack size 1000 - -5239.705755

- Reordering limits, stack size 10000, reordering limit 5 - -5081.342897

- Future cost estimation only with logprob, stack size 10,000, reordering limit 5 - -5081.342897

- Future cost estimation with logprob, tm, stack size 10,000, reordering limit 5 - -5081.342897

- Future cost estimation with logprob, lm, stack size 10,000, reordering reordering limit 5 - -5082.528174

- Future cost estimation with logprob, tm and lm, stack size 10,000, reordering limit 5 - -5082.528174

- Future cost estimation with logprob, tm, stack size 100,000, reordering limit 8 - -4969.320748

- Future cost estimation with logprob, lm, stack size 100,000, reordering limit 8 - -4993.047793

- Future cost estimation with logprob, tm and lm, stack size 100,000, reordering limit 8 - -4991.643550

- Future cost estimation with logprob, stack size 100,000, reordering limit 8 - -4991.643550





There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model

