# Direct Preference Optimisation Fine-Tuning on Anthropic HH-RLHF

> Created on: 17 July 2026
>
> Updated on: 24 July 2026

This note documents an implementation of the first Phase 3 stage: direct preference optimisation (DPO) ([Rafailov et al., 2023](#ref-rafailov2023)) of the supervised fine-tuning (SFT) policy on the pairwise preferences of `Anthropic/hh-rlhf`, replacing the reward model (RM) and proximal policy optimisation (PPO) stages of the classical reinforcement learning from human feedback (RLHF) pipeline with a single supervised loss. The stage is designed as the second arm of a controlled PPO-versus-DPO comparison: its data view, trainable capacity, and initialisation deliberately match the PPO stage documented in the PPO report.

The full source code can be found on [GitHub](https://github.com/nhan-dam/rlhf-course/blob/main/src/pipeline/dpo_lora_hh.py).

**Status note.** The implementation is complete and verified (compilation, configuration round-trip, and filter-semantics unit tests), but no training run has been executed yet. [Section 2](#2-exploratory-data-analysis), [Section 7](#7-results), and [Section 8](#8-reflections-and-next-steps) will be populated from the first run's artefacts; every design decision below is fixed independently of those results.

## 1. Background

The Kullback-Leibler (KL) penalised objective that PPO optimises numerically, maximising the RM score minus $\beta$ times the KL divergence from the reference policy $\pi_{\text{ref}}$, has an exact analytical maximiser: a softmax reweighting of $\pi_{\text{ref}}$ by the exponentiated reward. Inverting that closed form expresses the reward in terms of the policy it induces,

<span id="eq-implicit-reward"></span>

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x), \qquad (1)$$

where $Z(x)$ is a partition function that depends only on the prompt $x$. Substituting [(1)](#eq-implicit-reward) into the Bradley-Terry preference likelihood used to train the RM makes $Z(x)$ cancel (both responses share the prompt), leaving a loss defined on the policy alone,

<span id="eq-dpo-loss"></span>

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right], \qquad (2)$$

where $y_w$ and $y_l$ are the chosen and rejected responses and $\sigma$ is the logistic function. The two $\beta$-scaled log-ratios in [(2)](#eq-dpo-loss) are the **implicit rewards**: the quantity the RM stage learnt explicitly is here read off the policy itself. One supervised pass over the preference pairs therefore replaces both the RM fit and the PPO loop, with no rollout generation and no critic. What is given up is equally concrete: there is no explicit reward model to probe adversarially before training, and the optimisation is tied to the fixed preference dataset rather than to fresh on-policy samples.

The pipeline seam recorded in the PPO report's Section 1 applies unchanged here, and for the comparison that is a feature: both arms initialise from the same Dolly-trained SFT model and both learn from the same `Anthropic/hh-rlhf` preference distribution, so the seam cannot explain a quality difference between them.

## 2. Exploratory Data Analysis

Before training, the pairs are inspected with `eda_dpo_dataset.py`, so the two length caps of [Section 3.2](#32-length-filtering-matching-the-earlier-stages) rest on the data. DPO consumes the dataset whole, and three token lengths matter for every pair, each against a filter: the shared prompt against `max_prompt_tokens`, and each of the two sides, with the end-of-sequence (EOS) token appended, against `max_pair_tokens`. The script reports, to both the screen and a text file, the schema and splits, a random pair preview, the three length distributions, the per-cap and compound filtering trade-offs, a chosen-versus-rejected length-bias check (the DPO analogue of the RM's length-gaming trap, since the implicit reward in [(1)](#eq-implicit-reward) sums per-token log-ratios), and data-quality checks (empty sides, exact-duplicate pairs, and degenerate pairs with identical sides, which contribute a zero margin to [(2)](#eq-dpo-loss) by construction).

*The tables for this section will be inserted from the first run of the script over the full dataset.* Two anchors are already known from the earlier stages' EDA: the RM report measured pair-length percentiles with p95 at 562 and p99 at 866 tokens on the train split, and the PPO report measured 82.92% prompt retention at the 256-token prompt cap.

## 3. Implementation and Design Choices

### 3.1. The TRL v1 Application Programming Interface

`trl.DPOTrainer` accepts `Anthropic/hh-rlhf` directly: the dataset stores each pair as two full dialogues sharing a common prefix, which is TRL's 'implicit prompt' preference format. The trainer extracts the shared prompt internally and appends EOS to both sides itself, so no format preprocessing is required. One consequence of the v1 API matters for the design: its only length mechanism is `max_length`, which **truncates** (from the start or end per `truncation_mode`), and there is no filtering cap and no separate prompt cap. Truncation is not benign for preference pairs, because the two sides of an HH-RLHF pair differ mainly in the final assistant turn, so clipping tends to leave two near-identical prefixes whose preference label is uninformative. Length control is therefore implemented outside the trainer ([Section 3.2](#32-length-filtering-matching-the-earlier-stages)), and `max_length` is set only as a backstop that the pre-filter guarantees never binds.

### 3.2. Length Filtering Matching the Earlier Stages

`filter_pairs` drops, before the trainer sees them, every pair violating either cap. A pair survives only if its prompt (everything up to and including the final `\n\nAssistant:` marker, extracted with the PPO stage's own `extract_prompt`) fits `max_prompt_tokens` = 256, and both full dialogues, with EOS appended exactly as the trainer appends it, fit `max_pair_tokens` = 512. Both caps use filter semantics, never truncation, and both splits are filtered, the train split included.

The caps are not new numbers: 256 is the PPO stage's prompt cap and 512 is the RM stage's `max_length`, both fixed by those stages' EDA. Reproducing them here is a fairness requirement of the PPO-versus-DPO comparison, since it guarantees DPO trains on the same prompt distribution PPO optimised on and on the same length-admissible pairs the RM learnt from, so a quality difference between the two arms cannot be attributed to DPO having seen longer, shorter, or clipped data. The two caps overlap (a long pair usually has a long prompt), so the EDA's compound-filter table decomposes the joint retention into what each cap uniquely costs.

### 3.3. One Backbone for Policy and Reference

The policy is trained as a LoRA (low-rank adaptation, [Hu et al. (2022)](#ref-hu2022)) Parameter-Efficient Fine-Tuning (PEFT) model on the merged SFT backbone, and `ref_model=None` is passed to the trainer. As in the PPO stage, the trainer then recovers $\pi_{\text{ref}}$ by disabling the adapters, which is exact rather than approximate: a freshly initialised LoRA contributes $\Delta W = BA = 0$, so the adapter-disabled policy coincides with the merged SFT model, and the base stays frozen throughout. One copy of the Qwen2.5-0.5B backbone therefore serves both roles, halving the memory of the naive two-copy setup. During training, [`model_utils.CacheCleaner`](https://github.com/nhan-dam/rlhf-course/blob/main/src/common/model_utils.py) bounds PyTorch's reserved-memory pool, as in every stage; the SFT report's Section 5 presents the allocator analysis behind it.

### 3.4. Adapter Parity with the PPO Policy

The LoRA configuration is identical to the PPO policy adapter: rank 32, $\alpha = 64$, dropout 0.05, targets `q_proj` and `v_proj`. This is the second fairness requirement: the two arms of the comparison get the same trainable capacity attached to the same backbone, so neither can win by having more parameters to move. The $\alpha = 2r$ convention follows the earlier stages.

### 3.5. Learning Rate

The default is $5 \times 10^{-7}$, two to three orders of magnitude below the SFT and RM values. The implicit rewards in [(2)](#eq-dpo-loss) are differences of sequence-level log-probabilities, so small parameter steps translate into large margin movements, and larger learning rates reliably destabilise DPO training. Values between $10^{-7}$ and $10^{-6}$ are standard.

### 3.6. Beta, Loss Type, and Likelihood Displacement

$\beta$ prices drift from $\pi_{\text{ref}}$ inside the implicit reward, playing the role the KL coefficient plays in the PPO stage, but the two coefficients are not numerically comparable, so no attempt is made to match them. Instead $\beta$ (default 0.1) is the sweep axis of the comparison protocol in [Section 6](#6-comparative-evaluation-protocol). The loss defaults to the sigmoid form of [(2)](#eq-dpo-loss). A documented failure mode of that loss is **likelihood displacement**: the log-probabilities of chosen and rejected responses falling together, with the margin growing only because the rejected side falls faster, meaning probability mass is leaving the preference pairs entirely. The monitoring signal and the response are fixed in advance: if `logps/chosen` decreases alongside `logps/rejected`, switch `loss_type` to `'ipo'` ([Azar et al., 2024](#ref-azar2024)), whose bounded objective removes the incentive to push the margin without limit, or raise $\beta$. Both are configuration changes, not code changes.

### 3.7. Configuration-Driven Experiments and Run Tracking

The stage uses the same machinery as the other three: `DPOTrainingConfig` is parsed with `transformers.HfArgumentParser` from CLI overrides or a JSON file, the run label is the hash of the full resolved configuration, and each run writes `config_<label>.json` and `metrics_<label>.json` to its own results directory, joined and ranked by the shared `aggregate_metrics.py` (DPO runs rank by held-out implicit-reward accuracy). The $\beta$ sweep is therefore three commands differing in one flag, each landing in its own directory.

Unlike the experimental PPO trainer, `DPOTrainer` is a standard `Trainer` subclass, so the two capabilities the PPO stage lacks return here: an interrupted run resumes from its latest checkpoint (the unchanged config hashes to the same label, so the checkpoint directory is found automatically), and `load_best_model_at_end` keeps the checkpoint with the lowest evaluation loss, which is a sound selection metric because the DPO loss is monotone in the implicit-reward margin.

### 3.8. Post-Training Gate

Training-time evaluation scores a seeded 1,000-pair subsample of the length-filtered test split every 500 steps, enough precision for checkpoint selection at bounded cost. After training, the gate scores the **whole** filtered test split once and persists the result: the implicit-reward pairwise accuracy (the fraction of held-out pairs where the implicit reward ranks chosen above rejected), the mean margin, and the chosen and rejected log-probabilities, whose joint drift is the likelihood-displacement record of [Section 3.6](#36-beta-loss-type-and-likelihood-displacement). This mirrors the RM stage's two-role split of the same population, and the expectation band is the same 0.6 to 0.7 that gated the RM. Re-running a completed configuration recomputes exactly this block on the best checkpoint without retraining, via the same resume mechanism.

## 4. Training Configuration

The values below are the defaults. They are overridable from the command line or a JSON file (see [Section 3.7](#37-configuration-driven-experiments-and-run-tracking)).

| Hyperparameter | Value |
|---|---|
| Policy base | SFT model (merged), reference via adapter disabling |
| LoRA rank $r$ / scaling $\alpha$ / dropout | 32 / 64 / 0.05 (parity with the PPO policy) |
| LoRA target modules | `q_proj`, `v_proj` |
| $\beta$ | 0.1 (sweep axis: 0.05 / 0.1 / 0.3) |
| Loss type | sigmoid (switch to IPO on likelihood displacement) |
| Learning rate | $5 \times 10^{-7}$ |
| Epochs | 1 |
| Per-device batch size $\times$ gradient accumulation | $4 \times 4 = 16$ effective |
| Maximum prompt length | 256 tokens (filtering, not truncation) |
| Maximum pair length (each side, with EOS) | 512 tokens (filtering, not truncation) |
| Gradient checkpointing | disabled (default; configurable) |
| Precision | bfloat16 |
| Evaluation | 1,000 filtered test pairs / 500 steps; best checkpoint kept |

## 5. Training Diagnostics

- `rewards/accuracies`, the fraction of pairs whose implicit rewards are correctly ranked. It should climb from chance towards the 0.6 to 0.7 band; a flat curve at chance means the signal is too weak (raise the learning rate cautiously or check the data), while a rapid climb towards 1.0 suggests memorisation of the preference set rather than a generalisable ranking.
- `rewards/margins`, the mean implicit-reward margin, which should grow steadily. Margin growth with flat accuracy means existing correct pairs are being pushed further apart rather than new pairs being ranked correctly, which is the precursor of displacement.
- `logps/chosen` and `logps/rejected` together. Both falling is the likelihood-displacement signature of [Section 3.6](#36-beta-loss-type-and-likelihood-displacement), and the response (IPO or higher $\beta$) is a configuration change.
- The evaluation loss, which selects the checkpoint; divergence between falling train loss and rising evaluation loss is ordinary overfitting, expected within one epoch at this learning rate only if the rate is set too high.

## 6. Comparative Evaluation Protocol

The comparison against the PPO policy is designed before either arm is evaluated, so the protocol cannot drift towards whichever result looks better. It is implemented as a standing diagnostic, [`compare_policies.py`](https://github.com/nhan-dam/rlhf-course/blob/main/src/diagnostics/compare_policies.py), which takes the two run labels and writes `comparison_ppo_<ppo_label>_dpo_<dpo_label>.json` (every per-prompt record) and `.md` (summary table, head-to-head win rates, and a reading sample) under a shared results directory.

Four policies are evaluated, not two. Alongside the PPO and DPO arms, the SFT model anchors how much preference optimisation added on top of instruction tuning, and the raw pre-SFT base model (Qwen2.5-0.5B) anchors how much the entire pipeline added on top of the pre-trained model. The anchors are reference points rather than arms: neither judge is calibrated on raw base-model text (the RM was initialised from the SFT model and trained on HH-RLHF dialogue), so the base row is read qualitatively, not by its scores.

- **Evaluation prompts from the dedicated test split.** Generation prompts are extracted from the 8,552-row test split (with the same 256-token prompt filter), which no stage of either pipeline has trained on. The PPO stage's own 100 evaluation prompts were carved from the train split, which was sound for monitoring PPO in isolation but is unusable here, since those prompts sit inside DPO's training set.
- **Symmetric judging.** The RM is a biased judge (PPO was optimised against it directly), and DPO's implicit reward is biased in the mirror-image way. Both models' responses are therefore scored by both judges and reported as a 2$\times$2 table, with the RM stage's adversarial probes run against both models' outputs as a judge-independent check, and an external judge as tie-breaker if the two judges disagree.
- **Comparison at matched KL.** The PPO run is held fixed and DPO's $\beta$ is swept over 0.05, 0.1, and 0.3. Every run is placed on the reward-versus-KL plane, and the head-to-head pairs the PPO run with the DPO run of closest KL from $\pi_{\text{ref}}$. A single reward number at mismatched KL is not a comparison, since both methods can buy reward with drift.
- **Output diversity.** Distinct-bigram ratio and missing-EOS rate over the same generations (dependency-free stand-ins for self-BLEU), since preference optimisation can purchase margin with mode collapse or turn-closing failures.
- **Data budgets on record.** One DPO epoch sees every filtered training pair, whereas the PPO run consumed a 10,000-episode budget covering roughly 7.5% of its filtered prompts. The budgets cannot be meaningfully equalised, and are reported rather than hidden.

## 7. Results

*Pending the first training run and the $\beta$ sweep. This section will record, per run: the gate accuracy and margin over the full filtered test split, the final chosen and rejected log-probabilities, the training curves against the monitoring expectations of [Section 5](#5-training-diagnostics), and the comparative evaluation of [Section 6](#6-comparative-evaluation-protocol) against PPO run `e71b6d13`.*

## 8. Reflections and Next Steps

*Pending the results of [Section 7](#7-results). This section will record what the comparison with PPO settled and what it did not, whether $\beta$ or the loss type proved the binding choice, and any likelihood-displacement behaviour worth carrying into a later run.*

## 9. References

- <span id="ref-azar2024"></span>Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Rémi Munos, Mark Rowland, Michal Valko, Daniele Calandriello. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS 2024. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036).
- <span id="ref-hu2022"></span>Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- <span id="ref-rafailov2023"></span>Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn. *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.* NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
