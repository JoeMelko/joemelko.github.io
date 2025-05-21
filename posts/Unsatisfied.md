# The State Of Open Dataset Construction Is Deeply Unsatisfying

While many people find it more exciting to work on algorithms and optimizers, the lifeblood of AI models is the data they are trained on.

Simply put: *you need the right data to learn the right patterns*.

### Quick Acknowledgement

The work done on DataComp [1] is a wonderful breeding ground for new ideas. It would be great if more people did some hill climbing there :)


## The Crux of My Complaint

There are *so many* elements of uncertainty that stack when training large models. As a result, choosing primitives that minimize additional error is critical for any meta-algorithm like dataset selection/ordering.

**Most current methods don't use the right measure of utility.**

## Why Current Methods Are Insufficient

### Getting the simple stuff out of the way

I will focus on *slightly* more rigorous methods for the rest of this post, but will lead with the fact that

- yes, deduplication is helpful (sometimes)
- yes, language filtering is helpful (sometimes)
- yes, sequence/document/page length filtering is helpful (sometimes)
- yes, throwing out distorted/blurry images or videos is helpful (sometimes)

but if we consider the level of rigor applied to other avenues of research, these are *woefully* trivial baselines.

### On Classifiers and Embeddings

A natural starting point for large scale data filtering is classifiers. They are, perhaps, the most intuitive option when you step beyond the world of naive heuristics. They are also, currently, the best performing approach at scale with [2] providing strong results for multimodal filtering and an annoyingly simple FastText classifier is still the best open result filtering text corpora [3]. Additionally, there is related work on embeddings-based data curation [4,5] which selects data in order to maximize coverage of the embedding space and [6] which aims to match similarity to the target distribution of interest.

All of these methods are practically useful.

### So What's The Issue?

These methods are heavily dependent on what the practitioner determines to be "good" data. However, there are a litany of examples (such as [7]) that show human notions of data quality are extremely limited and often anti-correlates with downstream model performance.

They are based on the wrong primitives.

## The Best Way Forward

Since models often learn in weird, unintuitive ways, an optimal dataset selection algorithm should *measure* what the model is learning from different examples. Within this context, filtering can turn into an optimization problem over this utility measurement.

So what should this utility function be?

### Understanding Influence 

We can directly measure the influence that training on some example or set of examples has on an output / task of interest using Influence Functions [8]. They are defined as follows:

$$I(z, z_{test}) = -\nabla_{\theta} L(z_{test}, \theta^*)^T H_{\theta^*}^{-1} \nabla_{\theta} L(z, \theta^*)$$

where $\theta^*$ represents the optimal model parameters, $H_{\theta^*}$ is the Hessian matrix at $\theta^*$, $z$ is a training point, and $z_{test}$ is a test point we want to measure influence on. To start, there are 2 (very) apparent drawbacks of this approach:

- they are *very* expensive to compute
- they only contain information from the point in training where they are computed

As a result, there have been efforts to mitigate these problems by

- reducing dimension using random projections
- sampling from different model states 

like in [9] and [10]. There is also recent work [11], which makes 2 further contributions:

- computes influence over a period of training instead of a point in time
- iteratively retrains on better datasets --> better "period in training"

While these methods are still computationally prohibitive, they have already shown promise in constrained settings, which gives me confidence that we are converging on a much better primitive for *utility*.

## So What's Next

It seems natural to test previous *state-of-the-art* methods with influence-based utility as the scoring method. For example:

- influence-based classifier: training models to predict empirical influence instead of human notions of quality
- scaling laws: we can mitigate the computation inefficiencies if we can find weak to strong transfer conditions like [12] did for Hyperparameters
- pre-training a reasoner: optimizing data selection based on reasoning trace influence scores may lead to zero-shot reasoning capabilities or a better starting point for post-training

\+ many more things that are worth trying out.

## Until Next Time

If you got this far, let me know what you think + send me papers you think are cool!

Thanks for reading :)

## 

[1] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt. DataComp: In Search of the Next Generation of Multimodal Datasets. arXiv preprint arXiv:2304.14108 (2023).

[2] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, Vaishaal Shankar. Data Filtering Networks. arXiv preprint arXiv:2309.17425 (2023).

[3] Alon Albalak, Yanai Elazar, Sang Michael Xie, Shayne Longpre, Nathan Lambert, Xinyi Wang, Niklas Muennighoff, Bairu Hou, Liangming Pan, Haewon Jeong, et al. DataComp-LM: In Search of the Next Generation of Training Sets for Language Models. arXiv preprint arXiv:2406.11794 (2024).

[4] Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. SemDeDup: Data-Efficient Learning at Web-Scale Through Semantic Deduplication. arXiv preprint arXiv:2303.09540 (2023).

[5] Amro Abbas, Evgenia Rusak, Kushal Tirumala, Wieland Brendel, Kamalika Chaudhuri, Ari S. Morcos. Effective Pruning of Web-Scale Datasets Based on Complexity of Concept Clusters. arXiv preprint arXiv:2401.04578 (2024).

[6] Yiping Wang, Yifang Chen, Wendan Yan, Alex Fang, Wenjing Zhou, Kevin Jamieson, Simon Shaolei Du. CLIPLoss and Norm-Based Data Selection Methods for Multimodal Contrastive Learning. arXiv preprint arXiv:2405.19547 (2024).

[7] Gao, L., Biderman, S., Black, S., Golding, L., & Leahy, C. (2024). DataComp-LM: In Search of the Next Generation of Training Sets for Language Models. arXiv preprint arXiv:2401.10962.

[8] Pang Wei Koh, Percy Liang. Understanding Black-box Predictions via influence functions. arXiv preprint arXiv:1703.04730 (2017).

[9] Logan Engstrom, Axel Feldmann, Aleksander Madry. DsDm: Model-Aware Dataset Selection with Datamodels. arXiv preprint arXiv:2401.12926 (2024).

[10] Mengzhou Xia, Sadhika Malladi, Danqi Chen. LESS: Selecting Influential Data for Targeted Instruction Tuning. arXiv preprint arXiv:2402.04333 (2024).

[11] Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry. Optimizing ML Training with Metagradient Descent. arXiv preprint arXiv:2503.13751 (2025).

[12] Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. Advances in Neural Information Processing Systems 34, 2021.
