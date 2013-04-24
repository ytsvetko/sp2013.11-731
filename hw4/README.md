I implemented PRO reranker with l-2 regularized logistic regression (using https://github.com/redpony/creg), optimized towards Meteor, and following features: 

- original 3 features

- number of OOVs

- source/target lengths: delta and ratio

- punctuation and numbers in source/target sentences: delta and ratio

- Brown bigrams lexical features for target 

- Brown LM scores target sentence   

- BLEU and Meteor scores from dev/test target sentences to best translations produced by Moses-trained MT system with large LM


Brown-based features degraded performance and were discarded. 

Final feature weights:


2	***CATEGORICAL***	-1	1
-1	***BIAS***	-3.23278e-08
-1	bleu	-0.444914
-1	len_trgt	-0.345435
-1	p(e|f)	0.146904
-1	num_untranslated	2.19064
-1	len_src/trgt	0.845434
-1	meteor	-6.2158
-1	p_lex(f|e)	0.251633
-1	p(e)	0.114969
-1	punct:	-0.570199
-1	punct;	2.87431
-1	punct8	0.763545
-1	punct9	0.583959
-1	len_trgt/src	-0.247496
-1	punct?	-0.474427
-1	punct2	0.317305
-1	punct3	-0.769248
-1	punct0	-1.43342
-1	punct1	-0.295122
-1	punct6	0.493844
-1	punct7	0.172524
-1	punct5	-0.0420783
-1	punct(	0.19936
-1	punct)	0.642575
-1	punct.	0.752918
-1	punct-	-0.1301
-1	punct"	0.00410953
-1	punct!	1.45959
-1	punct'	-0.670173
-1	punct%	1.0798
