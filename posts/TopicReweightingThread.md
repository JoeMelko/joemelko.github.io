1/8
Are we really running out of data??? No. We're just not using it correctly.

The solution: let the model tell us what it needs!!!

[image]

2/8
Data curation has long been vibes + guesswork. Heuristics, classifiers, expensive ablations and grid searches. There are SO MANY degrees of freedom when working with data that methods often seem more like art than science.

But what if we had a way to directly optimize drastically more free parameters than before? Then can we be more principled?

3/8
YES!

First step: measuring (not guessing) how much training examples help us learn the thing(s) we care about.

Key idea: influence functions directly compute how training on one example affects the loss of another! 

But influence functions are COMPUTATIONALLY INFEASIBLE

4/8

Solution: efficient influence approximation

We introduce m‑TrackStar, a modification of TrackStar without the optimizer state, and with gradient clipping instead of L2-normalization. A simple, principled primitive.

4/8
Measuring utility on every example is still too slow. Luckily, we can work on clusters of examples instead.

The best part: it doesn't matter how you create these clusters! K-means, score based partitioning, language, document length or any other method you like - TerRIFIC is entirely agnostic.

5/8
Now all that is left to define the set of things you want the model to be good at and optimize:

(1) train a small model
(2) estimate cluster influence by aligning to the target set
(3) nudge logits.
(4) repeat

You don’t pick the data—the gradients do.

[Chart suggestion: Flow chart of the TerRIFIC outer loop]

6/8
Setup: ~37B token corpus -> ~12B subset for small models (DCLM‑Baseline), 10k FAISS k‑means clusters (Qwen3‑Embedding‑0.6B), target = OpenHermes 2.5. We train 411M models per meta step trained for 80% of Chinchilla.
[Chart suggestion: Cluster histogram or 2D embedding snapshot]

7/8
Results (@411M): Better NLL on Paloma, OpenThoughts‑114k, CodeAlpaca‑20k, OpenMathInstruct‑2; stable on DCLM‑Baseline; tiny recall tradeoff on WikiText. 

In only 2 meta iterations - grid‑searching 10k weights would be entirely infeasible!!!
[Chart suggestion: NLL vs meta‑iteration by task]

8/8
Scale transfer: We reintroduce held‑out tokens and train 1.4B parameter models on 28B tokens. Performance improvements are almost perfectly maintained - some even grow!

[Chart]

9/8
What about downsteam accuracy? These improve across the board on an 18 task subset of DCLM-Core, which we denote DCLM-Core-Clean. 

- we exclude AGI_EVAL_LSAT_AR, Commonsense_QA, BoolQ, and BB-CS-Algorithms due to high noise at these scales (411m models beating the 1.4B models with the same data mix!)
- results can still be found in the appendix of the post

10/8

Qualitatively: top clusters are math/chem/study resources; bottom clusters are pop‑culture slop + low‑context job posts + low‑level code (too hard for small models).
[]

11/8
TL;DR: If you can divide a dataset, you can learn how to sample it. It is dead simple: partition → pick targets → train small → update ratios → deploy at scale.

blog: https://joemelko.github.io/blog.html?post=TopicReweighting