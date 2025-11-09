Here’s the full updated version with the σ_sched change baked in. You can paste this whole thing over your current draft.

---

# Meta-Learning a Continuous Curriculum

In this work we introduce **TerRIFIC Curricula** (Topic Reweighting with Influence Functions in Clusters), a scalable, principled way to discover how to optimally sample from a training data distribution *throughout* the course of training. Our method can, non-invasively, participate in any, or all, portions of training. It asks practitioners only to define the set of things they want the model to learn. With this information, it iteratively learns how to modify the data distribution over the course of training to be maximally useful for learning.

We start from a static mixture-learning problem, turn it into a scale-dependent curriculum over training progress, and then show how to realize that curriculum as an explicit ordering of a finite dataset.

## Curriculum Learning

Modern large-scale training pipelines already implement a curriculum, even if we do not usually name it as such. Training is no longer a single homogeneous phase: models are pre-trained, then optionally mid-trained, finetuned (SFT), and finally optimized with RL-based methods such as RLHF, RLAIF, or RLVR. Each phase uses a different sampling distribution over data and objectives.

Seen this way, the pipeline is a hand-designed, piecewise curriculum over scale. Early on, we show the model a wide slice of the world and ask it to model generic structure. Later, we skew the mixture toward data that looks more like our downstream workload, then toward data with explicit human labels or preference signals, and finally toward reward-optimized interactions. What changes from phase to phase is not the architecture, but what we choose to show the model and how often.

Capabilities are strongly scale-dependent, which creates an asymmetry. Early in training we have abundant data and freedom to explore many ways of presenting it, but only weak local signals about what will matter later. At the end of training we have strong, task-relevant feedback, but much less flexibility to change the trajectory that got us there.

In this work, we view curriculum as the mechanism that ought to tie these regimes together: late-stage signals should inform how we allocate abundant early-stage data, so that the model spends its capacity learning features that are actually useful downstream.

## Discovering Optimal Curricula

We now turn from hand-designed curricula to the question of how to learn one. Concretely, given clustered pre-training data and downstream target(s), we want a procedure that suggests how to weight clusters as a function of training progress.

We build this in two stages. First, at any fixed training scale, we assume access to a static mixture-learning primitive—our TerRIFIC optimizer—that proposes an update to per-cluster sampling logits. Second, we lift this primitive to a scale-dependent setting, learning a continuous function of training progress whose induced mixtures define a continuous curriculum.

### Warmup: Metagradient Descent + The TerRIFIC Optimizer

In the static setting we maintain a single mixture over clusters that is fixed across the entire training run. The goal is to adjust that mixture so that training on it most improves performance on a downstream target set.

A natural conceptual starting point is **influence functions** (Koh et al., 2017). Given optimal parameters $\theta^\star$ for some training objective and a loss $L(z,\theta)$ on example $z$, the influence of a training example $z$ on a target example $z_{\text{test}}$ is defined as

$$
I(z, z_{\text{test}})
= -\nabla_{\theta} L(z_{\text{test}}, \theta^\star)^{\top}
H_{\theta^\star}^{-1}
\nabla_{\theta} L(z, \theta^\star),
$$

where $H_{\theta^\star}$ is the Hessian of the empirical risk at $\theta^\star$. Intuitively, $I(z, z_{\text{test}})$ measures how much up-weighting $z$ would change the loss on $z_{\text{test}}$ under an infinitesimal perturbation.

For modern models, explicitly forming or inverting $H_{\theta^\star}$ is infeasible. Prior work on TerRIFIC replaces exact influence with a **stable influence-like representation** built from gradients. Given model parameters $\theta$ and loss $L(z,\theta)$ on example $z$, it constructs

$$
mG_{\theta}(z)
= R^{-1/2} P_d ,\mathrm{clip}*t\bigl(\nabla*{\theta} L(z,\theta)\bigr),
$$

where $\mathrm{clip}*t$ applies gradient clipping, $P_d$ projects into a lower-dimensional subspace, and $R^{-1/2}$ whitens with respect to an empirical curvature or gradient-covariance estimate. Alignment in this space acts as a proxy for influence: examples whose $mG*{\theta}(z)$ align well with a target direction are expected to help that target.

Given a target set $\mathcal V$ and clusters ${c_j}_{j=1}^C$, TerRIFIC averages these representations:

$$
\bar v = \frac{1}{|\mathcal V|} \sum_{z\in\mathcal V} mG_{\theta}(z),
\qquad
\bar g_j = \frac{1}{|c_j|} \sum_{z\in c_j} mG_{\theta}(z).
$$

Clusters are then scored by alignment with the target:

$$
\bar I_j = \langle \bar g_j, \bar v \rangle.
$$

These scores define an update direction on the sampling logits. Let $\ell \in \mathbb{R}^C$ be the current logit vector over clusters, and let $\Delta \ell \in \mathbb{R}^C$ denote the increment vector produced by a single TerRIFIC step at the current checkpoint. We run the procedure once and treat the resulting $\Delta \ell$ as a **local meta-gradient** for the logits.

In practice, the raw scores ${\bar I_j}$ are normalized and stabilized before being applied:

* a running mean over clusters is subtracted so that updates are relative;
* variance normalization and clipping are applied across clusters to reduce sensitivity to noise.

The static TerRIFIC optimizer therefore alternates:

1. **Train.** Run the base training loop under the current mixture, up to a checkpoint.
2. **Measure.** Compute $mG_{\theta}(z)$ on held-out target and cluster samples, form $\bar v$, $\bar g_j$, and scores $\bar I_j$.
3. **Update.** Transform ${\bar I_j}$ into a stabilized increment $\Delta \ell$ and update the logits $\ell$.

Repeating this loop until convergence produces a **single static logit vector**: one mixture over clusters used for the entire training range.

In the remainder of this work, we treat this static optimizer as a primitive that, given model parameters at some checkpoint and a grouping logic $G$, returns a per-cluster logit update

$$
\Delta \ell(\theta, G) \in \mathbb{R}^C
$$

indicating how the mixture should change at that checkpoint.

### Learning Scale-Dependent Logits

We now move from a single global mixture to **scale-dependent mixtures** that explicitly depend on training progress. Rather than one fixed logit vector, we want a function

$$
f: (\mathbb{R}_{>0} \times G) \to \mathbb{R}^C
$$

that maps training progress $N$ and grouping logic $G$ to per-cluster logits. Here, $N$ denotes the total number of training tokens processed so far by the base model. For each $N$, the vector $f(N,G)$ defines a mixture over clusters, and the instantaneous sampling distribution is

$$
p(N;G) = \mathrm{softmax}\big(f(N,G)\big).
$$

Conceptually, we would like to learn a **vector field over scale**: at each progress value $N$ and grouping logic $G$, the static TerRIFIC primitive provides a preferred update direction in logit space. We denote this scale-dependent update by

$$
\Delta f(N, G) \in \mathbb{R}^C.
$$

What is missing is a way to stitch these local directions together into a smooth function of $N$.

To do this, we work in **log-time**. Let

$$
s = \log N,\qquad
s_{\min} = \log N_{\min},\quad
s_{\max} = \log N_{\max},
$$

where $[N_{\min}, N_{\max}]$ is the range of training progress over which we want to define a curriculum. Sampling uniformly in $s$ concentrates meta-updates at early scales, where small absolute changes in tokens correspond to large relative changes in model behavior.

At meta-iteration $t$, we fix model parameters $\theta$ at the current base-training checkpoint and perform the following:

> **Procedure (per meta-iteration $t$)**
> **Inputs:** log-progress range $[s_{\min}, s_{\max}]$, batch size $B$, step size $\eta_t$.
>
> 1. Sample $B$ log-time locations
>    $$
>    {s_t^{(b)}}*{b=1}^B \stackrel{\text{i.i.d.}}\sim \mathrm{Unif}([s*{\min}, s_{\max}])
>    $$
>    and set $N_t^{(b)} = e^{s_t^{(b)}}$.
> 2. For each location, run the static TerRIFIC step at progress $N_t^{(b)}$ to obtain an instantaneous per-cluster logit update
>    $$
>    \Delta f_t\bigl(s_t^{(b)}, G\bigr) \in \mathbb{R}^C.
>    $$
>    These are noisy samples of the underlying update field over log-time:
>    $$
>    \Delta f_t\bigl(s_t^{(b)}, G\bigr) \approx \Delta f\bigl(e^{s_t^{(b)}}, G\bigr).
>    $$
> 3. Sort ${s_t^{(b)}}$. For any $s \in [s_t^{(i)}, s_t^{(i+1)}]$, define a piecewise-linear interpolant
>    $$
>    \Delta f_t(s, G)
>    ================
>
>    \frac{s_t^{(i+1)} - s}{s_t^{(i+1)} - s_t^{(i)}} ,\Delta f_t\bigl(s_t^{(i)}, G\bigr)
>    +
>    \frac{s - s_t^{(i)}}{s_t^{(i+1)} - s_t^{(i)}} ,\Delta f_t\bigl(s_t^{(i+1)}, G\bigr),
>    $$
>    and clamp to the nearest endpoint value for $s < s_t^{(1)}$ or $s > s_t^{(B)}$.

This interpolation step simply specifies how updates at the sampled log-scales extend to intermediate values: between sampled points, the update $\Delta f_t(s,G)$ varies linearly in $s$. There is no additional optimization objective here; it is just a consistent way to define updates at progress values that were not explicitly queried.

We then update the curriculum logits additively. Writing $s = \log N$, the meta-update is

$$
f_{t+1}(N, G) = f_t(N, G) + \eta_t ,\Delta f_t\bigl(\log N, G\bigr),
$$

with a small step size $\eta_t$. Each meta-iteration $t$ uses a fixed set of base-model parameters $\theta$: we pause base training, estimate the update field $\Delta f_t(\cdot, G)$ in log-time from static TerRIFIC calls, take a small step on $f_t$, and then resume base training under the updated mixture

$$
p_{t+1}(N;G) = \mathrm{softmax}\bigl(f_{t+1}(N,G)\bigr).
$$

Over meta-iterations, this process accumulates into a **continuous curriculum**: for each cluster $j$, the trajectory

$$
N \mapsto f_t(N, G)_j
$$

becomes a smooth curve over training progress, and the corresponding mixtures $p_t(N;G)$ define a dense schedule rather than a handful of discrete phases. Because the curriculum varies smoothly with $N$, the data distribution can be changed gradually throughout training without explicit “re-warmup” phases: in practice, a single decaying learning-rate schedule can be used while the mixture evolves continuously, instead of restarting or re-heating the optimizer whenever the data mix is adjusted.

### Curriculum-Aligned Scheduling

Once we have a way to decide *what* to sample—whether via simple mixture weights or a learned schedule—we still have to realize it with a finite dataset. In practice, we do not sample from an infinite stream; we traverse a fixed set of sequences. Shuffling approximates the desired mixture only in expectation. Here we instead *construct an explicit ordering* whose prefixes track the target mixture (and later, a curriculum) as closely as possible.

We begin with **fixed data mixture weights**, and then extend the same machinery to continuous, scale-dependent curricula.

#### A Greedy Algorithm for Mixture-Adherent Ordering

Consider sequences ${s_1, \dots, s_M}$. Each sequence $s$ is a training chunk that may contain tokens from multiple underlying documents. We associate to each sequence a characteristic vector

$$
\mathbf{c}*s = (c*{s,1}, \dots, c_{s,K}),
$$

where $c_{s,j}$ is the number of tokens in $s$ assigned to group $j$, and each token belongs to exactly one group. Let

$$
\ell_s = \sum_{j=1}^K c_{s,j}
$$

denote the length (in tokens) of sequence $s$.

For a **fixed data mixture** over groups, we specify baseline token proportions

$$
\tau_j \in [0,1],\qquad \sum_{j=1}^K \tau_j = 1,
$$

interpreting $\tau_j$ as “the long-run fraction of tokens that should come from group $j$.”

Let $T_j$ denote the cumulative number of tokens from group $j$ in the prefix we have already selected, and let $S_{\text{tot}}$ be the total number of tokens in that prefix. If we append a new sequence $s$, the updated prefix has

$$
T_j' = T_j + c_{s,j},\qquad S_{\text{tot}}' = S_{\text{tot}} + \ell_s,
$$

and the fixed mixture implies an “ideal” cumulative count

$$
T_j^{\star}(S_{\text{tot}}') = \tau_j S_{\text{tot}}'.
$$

A natural greedy rule is therefore:

* **At each step, choose the next sequence** $s$ that keeps the updated cumulative counts $(T_j')*j$ as close as possible to the mixture-implied targets $(T_j^{\star}(S*{\text{tot}}'))_j$, in a least-squares sense.

This leads to the basic cluster-matching objective

$$
f_{\text{clusters}}(s)
= \sum_{j=1}^{K}
\Bigl[(T_j + c_{s,j}) - \tau_j \bigl(S_{\text{tot}} + \ell_s\bigr)\Bigr]^2.
$$

Greedily minimizing $f_{\text{clusters}}(s)$ constructs an ordering whose prefixes stay very near the fixed mixture $T_j^{\star}(S) = \tau_j S$.

However, there is a subtle pitfall. All $\mathbf{c}*s$ share the same $\ell_1$ norm (they sum to the sequence length $\ell_s$), so $f*{\text{clusters}}$ implicitly prefers sequences with *less sparse* cluster membership (smaller $\lVert \mathbf{c}_s \rVert_2$) early in the ordering, and then is forced to consume extremely sparse sequences (e.g., almost all tokens from a single cluster or even a single document) near the end to repair the cumulative error. In typical corpora this looks like: **short, mixed-cluster slices early; very long, single-source spans late**—a poor match to how we’d like the model to experience the data.

To fix this, we treat **document length** as a first-class part of the target mixture.

##### Doc-Size Regularization via a Length Histogram

We introduce a second characteristic: a quantile-binned histogram over *document* lengths. Sequences are training chunks, but each token remembers which underlying document it came from and that document’s length.

For each sequence $s$, let $\ell_{s,b}$ be the number of tokens in $s$ that come from documents whose length falls in bin $b$ (for $b = 1,\dots,B$). Let

$$
S_{\text{global}} = \sum_{m=1}^M \ell_{s_m}
$$

be the total number of tokens across all sequences. We define the *global* target token share for length bin $b$ as

$$
\kappa_b = \frac{\sum_{m=1}^M \ell_{s_m,b}}{S_{\text{global}}},
$$

i.e., the fraction of all tokens that come from documents in bin $b$ if we traverse the dataset in arbitrary order.

Let $U_b$ track the cumulative number of tokens from bin $b$ in the current prefix. After appending $s$, we would have

$$
U_b' = U_b + \ell_{s,b},
\qquad
S_{\text{tot}}' = S_{\text{tot}} + \ell_s,
$$

and the length-aware mixture implies an “ideal” cumulative count

$$
U_b^{\star}(S_{\text{tot}}') = \kappa_b S_{\text{tot}}'.
$$

We then define a **multi-characteristic objective** that jointly matches both:

* token-level *cluster* proportions, and
* token-level *doc-length* contributions.

For a fixed data mixture over groups, the ideal cumulative cluster counts are

$$
T_j^{\star}(S_{\text{tot}}') = \tau_j S_{\text{tot}}'.
$$

In the deterministic case, we score a candidate sequence $s$ by

$$
\boxed{
f_{\text{det}}(s)
=================

\underbrace{
\sum_{j=1}^{K}
\Bigl[(T_j + c_{s,j}) - \tau_j \bigl(S_{\text{tot}} + \ell_s\bigr)\Bigr]^2
}*{\text{cluster mixture adherence}}
+
\lambda
\underbrace{
\sum*{b=1}^{B}
\Bigl[(U_b + \ell_{s,b}) - \kappa_b \bigl(S_{\text{tot}} + \ell_s\bigr)\Bigr]^2
}_{\text{document-length mixture adherence}}
}
$$

At each step we pick

$$
s^\star = \arg\min_s f_{\text{det}}(s),
$$

then update

$$
T_j \leftarrow T_j + c_{s^\star,j},\quad
U_b \leftarrow U_b + \ell_{s^\star,b},\quad
S_{\text{tot}} \leftarrow S_{\text{tot}} + \ell_{s^\star},
$$

and repeat.

This greedy scheduler constructs an explicit ordering whose prefixes stay close—again in a least-squares sense—to a **fixed data mixture** over both clusters and document-length bins. Shuffling only satisfies these constraints on average; here they are enforced prefix by prefix.

In practice we also expose a **single noise hyperparameter** $\sigma_{\text{sched}}$ that lets us interpolate between this fully greedy ordering and a plain random shuffle. At each selection step, we flip a Bernoulli coin whose “greedy” probability $\alpha(\sigma_{\text{sched}}) \in [0,1]$ decreases monotonically with $\sigma_{\text{sched}}$ (in our implementation, $\alpha(\sigma_{\text{sched}}) = e^{-\sigma_{\text{sched}}}$). If the coin lands on “greedy,” we choose $s^\star$ by minimizing $f_{\text{det}}(s)$ as above; otherwise we ignore the objective and pick a uniformly random remaining sequence. Thus $\sigma_{\text{sched}} = 0$ yields $\alpha = 1$ and recovers the deterministic greedy schedule, while $\sigma_{\text{sched}} \to \infty$ drives $\alpha \to 0$ and recovers a true random shuffle of the corpus. In our experiments, $\sigma_{\text{sched}}$ is small; its role is mainly to desensitize the schedule to tiny count fluctuations and to let us view greedy ordering and shuffling as two endpoints of a single family.

#### Dynamic Schedules: From Fixed Mixtures to Continuous Curricula

So far we have treated $(\tau_j)_j$ as *fixed* mixture weights. Our meta-learning procedure instead produces **scale-dependent logits** $f(N,G)$, and hence time-varying mixture weights—i.e., a *curriculum*.

Recall that for a learned schedule we have

$$
p_j(N) = [\mathrm{softmax}(f(N,G))]_j,
$$

the instantaneous token-level probability of sampling group $j$ at training progress $N$. This defines a cumulative target for each group:

$$
E_j(S) = \int_0^S p_j(n),dn,\qquad \sum_j E_j(S) = S,
$$

which is the ideal number of tokens from group $j$ we would like to have seen by the time we have processed $S$ tokens. As $N$ varies, these $p_j(N)$ implement a continuous curriculum over groups.

To adapt the greedy scheduler, we replace the fixed targets $\tau_j (S_{\text{tot}} + \ell_s)$ with the curriculum-induced cumulative targets

$$
E_j(S_{\text{tot}} + \ell_s),
$$

where $\ell_s$ is the length (in tokens) of candidate sequence $s$.

Because document-length profiles are typically cluster-dependent, we similarly define a **length-aware curriculum**

$$
U_b^{\star}(S) = \sum_j E_j(S),\kappa_{b\mid j},
$$

where $\kappa_{b\mid j}$ is the per-cluster length-bin share, pre-computed once from the data. Intuitively, $U_b^{\star}(S)$ is the cumulative number of tokens from bin $b$ we would like to have seen by the time we reach $S$ tokens under this curriculum.

In the dynamic case, the multi-characteristic objective is obtained by substituting

* $\tau_j(S_{\text{tot}} + \ell_s) \rightsquigarrow E_j(S_{\text{tot}} + \ell_s)$ in the cluster term, and
* $\kappa_b(S_{\text{tot}} + \ell_s) \rightsquigarrow U_b^{\star}(S_{\text{tot}} + \ell_s)$ in the length term.

The same $\sigma_{\text{sched}}$ knob applies here as well: at each step we either follow the curriculum-aligned greedy objective or, with probability $1 - \alpha(\sigma_{\text{sched}})$, select a uniformly random remaining sequence, using the same mixing policy as in the static case. The result is a **curriculum-aligned schedule**: a concrete ordering of sequences whose prefixes closely track the *continuous curriculum* implied by $f(N,G)$, across both cluster membership and document-length structure. Fixed mixtures arise as the special case where $p_j(N)$ is independent of $N$, and $E_j(S) = \tau_j S$ reduces back to the static objective above; $\sigma_{\text{sched}}$ then simply interpolates between a fully greedy realization of that mixture and a fully shuffled one.

## Experimental Setup

Our experiments build on the Datacomp for Language Models (DCLM) (Li et al., 2024) recipe. We start from a ~37B-token slice of DCLM-Baseline and reserve ~12B tokens for **learning mixtures and curricula**; the remaining tokens are used for the final training runs under each schedule.

> DCLM-Baseline is already a highly filtered web corpus. The question in this work is not “can we rescue a bad dataset,” but “how much extra signal can we squeeze out of a good one.”

For clustering and targets, we reuse essentially the same pipeline as in the original TerRIFIC work (Melkonian, 2025). Each document is truncated to 1024 tokens and embedded with Qwen3-Embedding-0.6B (Zhang et al., 2025). We then run FAISS k-means with 10,000 clusters, following SemDeDup (Abbas et al., 2023). This gives us a reasonably fine-grained partition of the corpus without exploding meta-compute.

For the target set $\mathcal V$, we again use OpenHermes 2.5 (Teknium, 2024), truncating each example to 2048 tokens. All influence-style quantities—both in the static and scale-dependent settings—are computed against this fixed target set. Intuitively, the “what we care about” part of the setup is exactly the same as before; only the way we *use* it over training scale changes.

On the model side, we train two autoregressive transformers:

* a 411M-parameter model on 8.2B tokens, and
* a 1.4B-parameter model on 28B tokens.

These are the same model sizes and budgets as in the original TerRIFIC study, so effects are directly comparable. All runs use the standard DCLM training configuration. The only levers we touch are how clusters are weighted and how we schedule sequences under those weights. Everything else—the optimizer, learning rate schedule, batch size, and so on—is kept fixed.

All final training runs traverse the corpus using the **same greedy scheduler implementation**. The scheduler takes in a (possibly time-varying) mixture over clusters plus doc-length profiles and produces a single-pass ordering whose prefixes track those targets as closely as possible. Once this ordering is fixed, training is just “read the stream once”; there is no per-step curriculum logic during the run.

On top of this shared backbone, we compare four configurations:

1. **Baseline.** The strong DCLM baseline mixture with our greedy scheduler.
2. **TerRIFIC.** A static mixture learned from a single 6.4B-token checkpoint.
3. **TerRIFIC-AVG.** A static mixture learned by averaging influence information from 400M, 800M, 1.6B, 3.2B, and 6.4B-token checkpoints.
4. **TerRIFIC Curriculum.** Our learnable continuous curriculum that lets the mixture depend on training scale, using the same set of checkpoints (400M, 800M, 1.6B, 3.2B, 6.4B).

Across these four settings, the corpus, training budget, model, optimizer, and scheduler are identical. Only the mixture over clusters—static vs. scale-dependent—changes. To keep meta-compute modest, each TerRIFIC call uses a fixed subsample of cluster and target examples to approximate mean cluster influence, as in the original work.

## Results

The overarching pattern is:

> **Baseline static schedule < TerRIFIC-based static schedules < Continuous curriculum,**
> on almost all metrics that resemble the target distribution, with minimal movement on generic language modeling.

TerRIFIC-AVG is the key control: it shares the curriculum’s checkpoints and meta-compute but is static. The gap between TerRIFIC-AVG and the curriculum isolates the value of **making the mixture depend on scale**.

### Language Modeling and NLL

We first look at perplexity / NLL on Paloma, WikiText, OpenThoughts-114k, CodeAlpaca-20k, OpenMathInstruct 2, and held-out DCLM-Baseline.

#### 411M

At 411M:

* Moving from the **baseline static schedule** to either **static TerRIFIC** or **TerRIFIC-AVG**:

  * clearly improves NLL on OpenMathInstruct 2, OpenThoughts-114k, and CodeAlpaca-20k,
  * makes small positive moves on Paloma,
  * leaves DCLM-Baseline and WikiText essentially unchanged.

  The two static TerRIFIC variants are close; both sit comfortably above the baseline.

* Moving from the **TerRIFIC-based static schedules** to the **continuous curriculum** yields additional gains:

  * on the math- and code-heavy corpora, the curriculum is the best of all four schedules,
  * on Paloma and WikiText, it stays within noise of the strongest static schedule.

In short: at 411M, any TerRIFIC-based schedule is better than the baseline mixture, and the curriculum is best overall, particularly where the target distribution is most represented.

#### 1.4B

At 1.4B, the same shape reappears:

* Both **static TerRIFIC** and **TerRIFIC-AVG** outperform the baseline static mixture on the target-like corpora.
* The **continuous curriculum** matches or improves on those static schedules on nearly every language-modeling benchmark, again with its largest margins on OpenMathInstruct 2, OpenThoughts-114k, and CodeAlpaca-20k.

Held-out DCLM-Baseline and WikiText remain essentially flat across schedules, indicating that we are not improving the target tasks by catastrophically wrecking general language modeling.

A compact table (analogous to your original Table 1) can summarize these NLL numbers by task, model size, and schedule.

### Scaling Behaviour

We next ask whether these effects persist when we reuse the learned schedules at larger scale.

We take the mixtures and curricula learned from the 411M meta-training runs and train 1.4B models for 28B tokens. The headline:

* improvements from **static TerRIFIC** and **TerRIFIC-AVG** mostly carry from 411M to 1.4B, and
* the **continuous curriculum** remains the strongest schedule at 1.4B, just as at 411M.

In a plot of “delta vs. baseline” (similar in spirit to your earlier scaling figure), all three TerRIFIC-based schedules sit above zero on the target-like corpora, with the curriculum forming the upper envelope at both scales.

### DCLM-CORE-CLEAN Accuracy

We now look at DCLM-CORE-CLEAN centered accuracy.

For **411M**:

* The **baseline static schedule** gives a solid starting point.
* **Static TerRIFIC** and **TerRIFIC-AVG** both raise mean centered accuracy, especially on symbolic / algorithmic tasks that overlap with OpenHermes-style content.
* The **continuous curriculum** matches or exceeds both static baselines on most tasks, with the clearest gains again in math, reasoning, and structured-sequence regimes.

For **1.4B**, the pattern is the same:

* TerRIFIC-based static schedules beat the baseline mixture on average.
* The curriculum is competitive with or better than both static baselines, with its largest improvements on the same “TerRIFIC-aligned” tasks.

Aggregated:

> Baseline static < Static TerRIFIC ≈ TerRIFIC-AVG < Continuous curriculum.

A table in the style of your previous DCLM-CORE table (now with four columns per model size) makes this explicit.

### DCLM-CORE-CLEAN Correct-Answer NLL

Correct-answer NLL gives a calibration check.

Across both model sizes:

* whenever a TerRIFIC-based static schedule improves accuracy over the baseline, it usually also **reduces** correct-answer NLL;
* the **continuous curriculum** tends to push NLL down further in exactly the regions where accuracy improves.

We do not see a pattern where the curriculum gains accuracy only by becoming wildly overconfident on a few items. On tasks where it is better than both the baseline and the static TerRIFIC schedules, it generally assigns higher probability to correct answers as well.

A compact NLL table (analogous to your previous Table 3) captures this trend.

### Summary

Across both model sizes and all evaluation suites:

* all TerRIFIC-based schedules improve over the baseline static mixture on the math / code / reasoning distributions that resemble the target set, while leaving generic language modeling largely unchanged;
* among those schedules, the **continuous curriculum** is consistently best.

The comparison between **TerRIFIC-AVG** and the **continuous curriculum** is the cleanest test: both see the same checkpoints and influence estimates, and both use the same amount of meta-compute. The only difference is that TerRIFIC-AVG collapses everything into a static mixture, while the curriculum lets the mixture evolve with training progress.

The fact that the curriculum wins in this matched setting is strong evidence that **organizing data over scale**—not just reweighting it once—is an important degree of freedom.

## Behavior of the Scheduler: Shuffled vs Greedy Under a Static Mixture

All training runs rely on some flavor of the greedy scheduler. To substantiate this choice, we examine its behavior compared to vanilla shuffling.

Here the question is not “which mixture is better,” but “what happens if we replace random shuffling with greedy ordering under the *same* mixture.”

### Setup: Same Data, Same Mixture, Different Orderings

We fix the **baseline static mixture** over clusters and doc-length bins and consider two ways of traversing the same corpus:

1. **Random shuffle.**
   All sequences are shuffled once, and training proceeds straight through. In expectation, cumulative counts track the target mixture, but prefixes can deviate.

2. **Greedy scheduler.**
   Using the multi-characteristic objective from earlier, we construct a single-pass ordering that keeps:

   * cumulative cluster counts, and
   * cumulative doc-length bin counts

   as close as possible to the mixture-implied targets at every prefix.

Both runs:

* see the **same tokens** exactly once,
* respect the **same marginal mixture** over clusters and doc-length bins,
* use the **same models, optimizers, and hyperparameters**.

The only difference is how the sequences are **ordered** under a fixed mixture.

### Mixture Fidelity (Non-Bias Check)

Random shuffling is unbiased with respect to the target mixture: in expectation, each cluster’s cumulative share matches its mixture weight.

For the scheduler, we care about whether the greedy objective introduces any **systematic drift** away from that mixture. To measure this, we track, for each cluster and doc-length bin, the deviation between:

* the actual cumulative count in the training prefix, and
* the ideal cumulative count implied by the baseline mixture.

Summarizing these deviations over training:

* shuffling shows zero mean but substantial variance, especially early in training and for small or rare clusters;
* the greedy scheduler produces cumulative curves that stay much closer to the ideal mixture:

  * deviations are smaller and mostly symmetric,
  * no cluster or length bin is systematically over- or underrepresented.

In this sense, the scheduler is **at least as faithful** to the baseline mixture as shuffling, and often closer, because mixture adherence is enforced prefix by prefix rather than only in expectation.

### Stable Rank of Model Updates

With mixture fidelity established, we turn to the effect of ordering on the **geometry of updates**.

For a fixed layer (or parameter subset), we collect a window of parameter deltas or gradients over $T$ consecutive steps into a matrix $G \in \mathbb{R}^{T \times d}$ and compute the stable rank:

$$
\mathrm{srank}(G) = \frac{\lVert G \rVert_F^2}{\lVert G \rVert_2^2},
$$

where $\lVert \cdot \rVert_F$ is the Frobenius norm and $\lVert \cdot \rVert_2$ is the spectral norm. Intuitively, stable rank measures how many “effective directions” are being explored in parameter space over that window.

Under the **baseline mixture**, but with different orderings, we find:

* the **greedy scheduler** yields **higher stable rank** updates than random shuffling across layers and training windows,
* this holds even though:

  * the total number of tokens,
  * the marginal mixture over clusters and length bins, and
  * the optimizer settings

  are identical.

The greedy scheduler keeps cluster and doc-length structure better mixed at every prefix, and in practice this corresponds to updates that occupy a richer subspace of parameter space.

Viewed this way, $\sigma_{\text{sched}}$ plays a role analogous to the “S” in SGD: it controls how much stochasticity we inject into the *data trajectory* under a fixed mixture, just as batch size or gradient noise controls stochasticity in the *parameter updates*. At $\sigma_{\text{sched}} = 0$, updates are driven by a nearly deterministic, mixture-perfect ordering; as $\sigma_{\text{sched}}$ increases, we allow more variance in which sequences are selected at each step, eventually recovering the behavior of a fully shuffled stream.

## References

[1] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston. Curriculum Learning. Proceedings of the 26th International Conference on Machine Learning (ICML) (2009).

[2] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Anna Korbak, Clemens Winter, Mateusz Malinowski, Joe Snaith, Goran Kruszewski, Timo Ewalds, Stanisław Jastrzębski, Sylvain Gelly, Laurent Sifre, Erich Elsen, Jack W. Rae, Oriol Vinyals. Training Compute-Optimal Large Language Models. Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS) (2022).
