While it can often seem more exciting to work on algorithms and optimizers, the lifeblood of AI models is the data they are trained on.

Simply put: *you need the right data to learn the right patterns*.

<h2 id="high-level-framing">High-Level Framing</h2>

The goal of dataset selection is to find a subset of data that will lead to good model performance. Ideally, this can be framed as an optimization problem:

$$
\max_{S \subseteq D} U(\theta_S^*) \quad \text{s.t.} \quad |S| \leq k
$$

where $S$ is our selected subset, $D$ is the full dataset, $\theta_S^*$ represents a model trained on subset $S$ with parameters $\theta$, $U(\cdot)$ is some utility function, and $k$ is our target dataset size due to compute constraints.

*Note: The above formulation is idealistic. In practice **most** algorithms use $\max_{S \subseteq D}\sum_{x \in S} U(x)$ as computing $\theta_S^*$ is prohibitively expensive.*

<h2 id="the-crux-of-my-complaint">The Crux of My Complaint</h2>

There are *so many* elements of uncertainty that stack when training large models. As a result, choosing primitives that minimize additional error is critical for any meta-algorithm like dataset selection.

**Most existing methods don't pick the right utility function.**

<h2 id="a-quick-note-on-timing">A Quick Note On Timing</h2>

Rigorous evaluation of large-scale data curation methods has traditionally been computationally prohibitive. However, as the bottleneck shifts from compute constraints toward data constraints, it becomes increasingly important to maximize what we learn from the data we already have.

More concisely: *trading compute for data quality is becoming a better and better deal.*

<h2 id="why-current-methods-are-insufficient">Why Current Methods Are Insufficient</h2>

<h3 id="getting-the-simple-stuff-out-of-the-way">Getting the simple stuff out of the way</h3>

I will focus on *slightly* more rigorous methods for the rest of this post, but will lead with the fact that

- yes, deduplication is helpful (sometimes)
- yes, language filtering is helpful (sometimes)
- yes, sequence/document/page length filtering is helpful (sometimes)
- yes, throwing out distorted/blurry images or videos is helpful (sometimes)

but if we consider the level of rigor applied to other avenues of research, these are *woefully* trivial baselines.

<h3 id="on-classifiers-and-embeddings">On Classifiers and Embeddings</h3>

A natural starting point for large scale data filtering is classifiers. They are, perhaps, the most intuitive option when you step beyond the world of naive heuristics. They are also, currently, the best performing approach at scale with [2] providing strong results for multimodal filtering and an annoyingly simple FastText classifier is still the best open result for filtering text corpora [3]. Additionally, there is related work on embeddings-based data curation [4,5], which selects data in order to maximize coverage of the embedding space and [6] which aims to match similarity to the target distribution of interest.

All of these methods are practically useful.

<h3 id="so-whats-the-issue">So What's The Issue?</h3>

These methods are heavily dependent on what the practitioner determines to be "good" data. However, there are a litany of examples (such as [7]) that show human notions of data quality are extremely limited and often anti-correlate with downstream model performance.

<h2 id="the-best-way-forward">The Best Way Forward</h2>

It is far simpler to identify the things we would like a model to be able to do than to guess what will help get us there. Since models often learn in unintuitive ways, an optimal dataset selection algorithm should *measure* what the model is learning from different examples in relation to what we want it to know. Luckily, we can do just that.

*This setup loosely resembles RL, while being at a different level of abstraction: episodes correspond to training runs, and updates correspond to adjustments in data weighting.*

<h3 id="understanding-influence">Understanding Influence</h3>

We can directly compute the effect that training on some example, or set of examples, has on an output / task of interest using Influence Functions [8]. They are defined as follows:

$$
I(z, z_{test}) = -\nabla_{\theta} L(z_{test}, \theta^*)^T H_{\theta^*}^{-1} \nabla_{\theta} L(z, \theta^*)
$$

where $\theta^*$ represents the optimal model parameters, $H_{\theta^*}$ is the Hessian matrix at $\theta^*$, $z$ is a training point, and $z_{test}$ is a test point we want to measure influence on. To start, there are two very apparent limitations of this approach:

- they are *very* expensive to compute
- they only contain information from the point in training where they are computed

As a result, there have been efforts to mitigate these problems by:

- reducing dimensionality using random projections
- sampling across different model states 

as demonstrated in [9,10,11]. Recent work [12] provides two additional contributions:

- computes influence over a period of training instead of at a single step
- iterative retraining on better datasets yields updates on better "periods in training"

While these methods are still computationally prohibitive, they have already shown promise in constrained settings, which gives me confidence that we are converging on a much better primitive for *utility*.

<h2 id="so-whats-next">So What's Next</h2>

There is still plenty of work needed to mitigate the limitations of current gradient-based approaches, such as:

- <u>**designing minimally lossy dimensionality reduction**</u>: pseudo-random projections carry nice guarantees, but in practice we should be able to construct or learn a better low-rank representation
- modeling temporal influence: predict how influence adapts during training [13] can aid methods like [12] towards the eventual goal of scaling laws and weak-to-strong transfer, like [14] did for hyperparameters

It also seems natural to test hybrid approaches that combine the efficiency of prior methods with influence as the underlying utility measure, for example:

- training influence-based classifiers: create models to predict empirical influence rather than relying on human notions of quality (early work in [15])
- <u>**experimenting with tiered approaches**</u>: assigning influence to clusters or sources instead of to individual samples or documents

Finally, there are some more speculative directions that could be worth exploring, like:

- pre-training a reasoner: optimizing data via reasoning-trace influence may lead to zero-shot reasoning capabilities or a better base for post-training
- guiding synthetic data generation: using influence scores to greedily fill capability gaps in training data

*Note: I am actively working on the ideas that are underlined + bold and will share more soon!*

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

[13] Jiachen T. Wang, Dawn Song, James Zou, Prateek Mittal, Ruoxi Jia. Capturing the Temporal Dependence of Training Data Influence. arXiv preprint arXiv:2412.09538 (2024). 

[14] Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. Advances in Neural Information Processing Systems 34, 2021.

[15] Zichun Yu, Fei Peng, Jie Lei, Arnold Overwijk, Wen‑tau Yih, Chenyan Xiong. Data‑Efficient Pretraining with Group‑Level Data Influence Modeling. arXiv preprint arXiv:2502.14709 (2025).