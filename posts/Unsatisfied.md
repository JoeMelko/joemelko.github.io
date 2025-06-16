## Table of Contents
- <span style="color: #333333">[High-Level Framing](#high-level-framing)</span>
- <span style="color: #333333">[The Crux of My Complaint](#the-crux-of-my-complaint)</span>
- <span style="color: #333333">[A Quick Note On Timing](#a-quick-note-on-timing)</span>
- <span style="color: #333333">[Why Current Methods Are Insufficient](#why-current-methods-are-insufficient)</span>
  - <span style="color: #333333">[Getting the simple stuff out of the way](#getting-the-simple-stuff-out-of-the-way)</span>
  - <span style="color: #333333">[On Classifiers and Embeddings](#on-classifiers-and-embeddings)</span>
  - <span style="color: #333333">[So What's The Issue?](#so-whats-the-issue)</span>
- <span style="color: #333333">[The Best Way Forward](#the-best-way-forward)</span>
  - <span style="color: #333333">[Understanding Influence](#understanding-influence)</span>
- <span style="color: #333333">[So What's Next](#so-whats-next)</span>


While many people find it more exciting to work on algorithms and optimizers, the lifeblood of AI models is the data they are trained on.

Simply put: *you need the right data to learn the right patterns*.

## High-Level Framing {#high-level-framing}

The goal of dataset selection is to find a subset of data that will lead to good model performance. Ideally, this can be framed as an optimization problem:

$$
\max_{S \subseteq D} U(\theta_S^*) \quad \text{s.t.} \quad |S| \leq k
$$

where $S$ is our selected subset, $D$ is the full dataset, $\theta_S^*$ represents a model trained on subset $S$ with parameters $\theta$, $U(\cdot)$ is some utility function, and $k$ is our target dataset size due to compute constraints.

*Note: The above formulation is idealistic. In practice **most** algorithms use $\max_{S \subseteq D}\sum_{x \in S} U(x)$ as computing $\theta_S^*$ is prohibitively expensive.*

## The Crux of My Complaint {#the-crux-of-my-complaint}

There are *so many* elements of uncertainty that stack when training large models. As a result, choosing primitives that minimize additional error is critical for any meta-algorithm like dataset selection.

**Most existing methods don't pick the right utility function.**

## A Quick Note On Timing {#a-quick-note-on-timing}

Rigorous evaluation of large-scale data curation methods has traditionally been computationally prohibitive. However, as the bottleneck shifts from compute constraints toward data constraints, it becomes increasingly important to maximize what we learn from the data we already have.

More concisely: *trading compute for data quality is becoming a better and better deal.*

## Why Current Methods Are Insufficient {#why-current-methods-are-insufficient}

### Getting the simple stuff out of the way {#getting-the-simple-stuff-out-of-the-way}

I will focus on *slightly* more rigorous methods for the rest of this post, but will lead with the fact that

- yes, deduplication is helpful (sometimes)
- yes, language filtering is helpful (sometimes)
- yes, sequence/document/page length filtering is helpful (sometimes)
- yes, throwing out distorted/blurry images or videos is helpful (sometimes)

but if we consider the level of rigor applied to other avenues of research, these are *woefully* trivial baselines.

### On Classifiers and Embeddings {#on-classifiers-and-embeddings}

A natural starting point for large scale data filtering is classifiers. They are, perhaps, the most intuitive option when you step beyond the world of naive heuristics. They are also, currently, the best performing approach at scale with [2] providing strong results for multimodal filtering and an annoyingly simple FastText classifier is still the best open result for filtering text corpora [3]. Additionally, there is related work on embeddings-based data curation [4,5], which selects data in order to maximize coverage of the embedding space and [6] which aims to match similarity to the target distribution of interest.

All of these methods are practically useful.

### So What's The Issue? {#so-whats-the-issue}

These methods are heavily dependent on what the practitioner determines to be "good" data. However, there are a litany of examples (such as [7]) that show human notions of data quality are extremely limited and often anti-correlate with downstream model performance.

## The Best Way Forward {#the-best-way-forward}

It is far simpler to identify the things we would like a model to be able to do than to guess what will help get us there. Since models often learn in unintuitive ways, an optimal dataset selection algorithm should *measure* what the model is learning from different examples in relation to what we want it to know. Luckily, we can do just that.

*This setup loosely resembles RL, while being at a different level of abstraction: episodes correspond to training runs, and updates correspond to adjustments in data weighting.*

### Understanding Influence {#understanding-influence}

We can directly compute the effect that training on some example, or set of examples, has on an output / task of interest using Influence Functions [8]. They are defined as follows:

$$
I(z, z_{test}) = -\nabla_{\theta} L(z_{test}, \theta^*)^T H_{\theta^*}^{-1} \nabla_{\theta} L(z, \theta^*)
$$

where $\theta^*$ represents the optimal model parameters, $H_{\theta^*}$ is the Hessian matrix at $\theta^*$, $z$ is a training point, and $z_{test}$ is a test point we want to measure influence on. To start, there are two very apparent limitations of this approach:

- they are *very* expensive to compute
- they only contain information from the point in training where they are computed

As a result, there have been efforts to mitigate these problems by

- reducing dimensionality using random projections
- sampling across different model states 

as demonstrated in [9,10,11]. Recent work [12] provides two additional contributions:

- computes influence over a period of training instead of at a single step
- iterative retraining on better datasets yields updates on better "periods in training"

While these methods are still computationally prohibitive, they have already shown promise in constrained settings, which gives me confidence that we are converging on a much better primitive for *utility*.

## So What's Next {#so-whats-next}

It seems natural to test previous *state-of-the-art* methods with influence-based utility metrics. For example:

- influence-based classifiers: training models to predict empirical influence rather than relying on human notions of quality (early work in [13])
- scaling laws: we can mitigate the computation inefficiencies if we can find weak to strong transfer conditions like [14] did for hyperparameters
- pre-training a reasoner: optimizing data selection based on reasoning trace influence scores may lead to zero-shot reasoning capabilities or a better starting point for post-training

\+ many more things that are worth trying out.

Thanks for reading :)

---

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

[11] Qirun Dai, Weijia Shi, Jiacheng Ye, Rongzhi Zhang, Xinyu Dai, Yulan He, Shujian Huang, Jiajun Chen. Improving Influence-based Instruction Tuning Data Selection for Balanced Learning of Diverse Capabilities. arXiv preprint arXiv:2501.12147 (2025).

[12] Logan Engstrom, Andrew Ilyas, Benjamin Chen, Axel Feldmann, William Moses, Aleksander Madry. Optimizing ML Training with Metagradient Descent. arXiv preprint arXiv:2503.13751 (2025).

[13] Zichun Yu, Fei Peng, Jie Lei, Arnold Overwijk, Wen‑tau Yih, Chenyan Xiong. Data‑Efficient Pretraining with Group‑Level Data Influence Modeling. arXiv preprint arXiv:2502.14709 (2025).

[14] Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. Advances in Neural Information Processing Systems 34, 2021.