I implemented reordering, distortion limits, lookup tables with future cost estimation by TM score, by LM score. 

Efficient implementation of keys for partially translated sentences: bitmasks.  

A*-like search heuristics: find best score for smaller stack size, then increase stack size and don't insert to stacks hypotheses that have lower probability than current best hypothesis. Iteratively increase stacks until stack size reaches predefined limit.  
 
Finally, I combine best outputs from several runs. 

