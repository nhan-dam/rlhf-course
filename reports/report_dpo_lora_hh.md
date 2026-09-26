# Direct Preference Optimisation Fine-Tuning on Anthropic HH-RLHF

> Created on: 17 July 2026
>
> Updated on: 26 September 2026

This note documents an implementation of the first Phase 3 stage: direct preference optimisation (DPO) ([Rafailov et al., 2023](#ref-rafailov2023)) of the supervised fine-tuning (SFT) policy on the pairwise preferences of `Anthropic/hh-rlhf`, replacing the reward model (RM) and proximal policy optimisation (PPO) stages of the classical reinforcement learning from human feedback (RLHF) pipeline with a single supervised loss. The stage is designed as the second arm of a controlled PPO-versus-DPO comparison: its data view, trainable capacity, and initialisation deliberately match the PPO stage documented in the PPO report.

The full source code can be found on [GitHub](https://github.com/nhan-dam/rlhf-course/blob/main/src/pipeline/dpo_lora_hh.py).

**Status note.** Implementation, exploratory data analysis and both hyperparameter sweeps are complete, seven runs in all. The selected arm is `75047d16`, at a learning rate of $10^{-4}$ and $\beta = 0.1$, promoted to the canonical `dpo-model/` path. [Section 7](#7-results) reports it, and [Section 9](#9-appendix-hyperparameter-sweeps) reports every run and the selection. The comparison against PPO in [Section 6](#6-comparative-evaluation-protocol) has not yet been run.

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

The figures below are from a run of the script over both splits in full. The corpus is the one the RM stage analysed, i.e. a `train` split of 160,800 pairs and a dedicated `test` split of 8,552, so this section reports what is new for DPO and anchors everything else to the RM report, Section 2, rather than repeating it. The two splits are treated differently on purpose. The training split gets the full analysis including a random preview. The test split is the evaluation population, i.e. the in-training evaluation subsample, the post-training gate, and the comparison prompts of [Section 6](#6-comparative-evaluation-protocol), so it gets every aggregate check and no preview, since reading evaluation examples is what a protocol fixed in advance should not do.

### 2.1. Three Lengths, and a Tokenisation Check

The RM saw two lengths per pair. DPO sees three, because the prompt is split off and capped on its own. The chosen and rejected columns reproduce the RM report's Table 1 to the token at every percentile (p50 168 and 163, p95 535 and 537, p99 824 and 835, maxima 1,966 and 2,105), which is the check that the pipeline's tokenisation, i.e. marker split plus EOS, matches the RM's view of the same pairs. Only the prompt column is new.

<a id="tab-dpo-prompt-len"></a>

| Percentile | Train prompt | Train `max(pair)` | Test prompt | Test `max(pair)` |
|---|---|---|---|---|
| p50 | 100 | 189 | 103 | 191 |
| p75 | 203 | 302 | 205 | 302 |
| p90 | 323 | 454 | 325 | 454 |
| p95 | 421 | 562 | 423 | 565 |
| p99 | 688 | 866 | 722 | 892 |
| p99.9 | 1306 | 1579 | 1419 | 1670 |
| max | 1896 | 2105 | 1868 | 1928 |

Table 1: Token-length percentiles of the shared prompt and of the longer side per pair with EOS, in each split, tokenised with the SFT tokeniser after the marker split and EOS append.

The prompt is long-tailed in the same way the pairs are, with a median of 100 tokens and a 99th percentile near 700, so the 256-token prompt cap sits between p75 and p90 and is not a rounding decision. The two splits coincide to within a few tokens through p95 and diverge only in the extreme tail, which is the same agreement the RM report found for the pair lengths.

### 2.2. The Two Caps and Their Compound Effect

Each cap is inherited from an earlier stage, i.e. the prompt cap of 256 from PPO and the pair cap of 512 from the RM, and each is a filter, never a truncation. Applied alone to the training split they reproduce the earlier stages' retention exactly: 133,331 pairs (82.92%) under the prompt cap, the PPO figure, and 149,566 (93.01%) under the pair cap, the RM figure. The test split tracks the training split within 0.4 percentage points at every candidate value of either cap, so caps chosen on train transfer to the evaluation population without adjustment. What the earlier EDAs could not report is the caps' intersection.

<a id="tab-dpo-compound"></a>

| Outcome at the selected caps (prompt 256, pair 512) | Train pairs | Train % | Test pairs | Test % |
|---|---|---|---|---|
| Kept (both caps pass) | 133,013 | 82.72% | 7,039 | 82.31% |
| Dropped by prompt cap only | 16,553 | 10.29% | 913 | 10.68% |
| Dropped by pair cap only | 318 | 0.20% | 16 | 0.19% |
| Dropped by both | 10,916 | 6.79% | 584 | 6.83% |
| Kept and sides share the final-turn prompt | 132,924 | 82.66% | 7,033 | 82.24% |

Table 2: Decomposition of each split under the compound filter at the selected caps, with the last row applying the prompt-sharing exclusion on top.

The prompt cap is the binding constraint: it accounts for 17.08% of the training split on its own, while the pair cap uniquely removes 318 pairs, 0.20 percentage points. So RM parity is nearly free and PPO parity is what DPO pays for, since without the prompt cap DPO could train on 93% of the corpus. That price is accepted deliberately. The comparison of [Section 6](#6-comparative-evaluation-protocol) is only clean if both arms saw the same population, and the pairs the prompt cap removes are systematically the long multi-turn dialogues, so an arm trained with them would differ from PPO in data as well as in method. The last row of [Table 2](#tab-dpo-compound) is the one non-length exclusion. The prompt is cut at the chosen side's final `\n\nAssistant:` marker, and in 89 kept training pairs and 6 kept test pairs the rejected side does not begin with that prompt, because the two transcripts diverge at an earlier assistant turn. Those are two conversations rather than two responses to one prompt, so the pipeline drops them rather than mis-split them. The RM stage, which scores whole dialogues, kept them. The DPO training set is therefore 132,924 pairs, and the filtered test split is 7,033 pairs, against the RM's 7,952 under its single cap. [Section 3.8](#38-post-training-gate) divides those into a 1,000-pair in-training evaluation subsample and the 6,033-pair gate.

### 2.3. Chosen-versus-Rejected Length Bias

The check matters more for DPO than it did for the RM. The RM scored a sequence with a scalar head, so length could only reach the score through what the head learned. DPO's implicit reward in [(1)](#eq-implicit-reward) is a *sum* of per-token log-ratios, so response length enters the margin in [(2)](#eq-dpo-loss) directly, and a corpus in which the preferred side is reliably longer would hand the policy a shortcut that costs nothing. Neither split offers one. In the training split `chosen` is the longer side in 50.9% of pairs and `rejected` in 47.1%, with 2.0% tied, a mean difference of +3.4 tokens and a median of +1. The test split reads 51.7%, 46.3%, 2.0%, +2.9 and +2. The training figure is the RM report's finding restated, since it is the same data, and the consequence is the same: any length growth observed in the trained policy is something the optimisation introduced, not something the data rewarded. The test figure matters separately, because the gate is scored there: a length-driven implicit reward cannot pass the gate on the strength of the evaluation pairs' lengths.

### 2.4. Data Quality and a Tokenisation Observation

The training-split counts are the RM's, since the split is the same: 0 empty sides, 740 pairs (0.46%) with identical sides, and 0 exact duplicates. Identical pairs are inert under DPO in a stronger sense than under the Bradley–Terry loss: the margin is zero by construction, and since the two sides are the same text their gradient contributions cancel exactly, so the 740 pairs cost forward passes and nothing else. They are kept for parity with the RM's data view. The test split has 36 identical pairs (0.42%). Those are not inert. An identical pair can only tie, and the gate counts a tie as a failure under its strict inequality, so at most 36 of the 7,033 filtered test pairs, 0.5%, are failures no policy can avoid. That bounds the gate's ceiling, not its floor, and is far below the 0.6 to 0.7 band, but it is the number to subtract before reading any gate accuracy as a rate of genuine mis-rankings.

The two splits do not leak into each other. No test dialogue and no test pair appears anywhere in the training split. 72 test pairs (0.84%) share their prompt with a training pair, which is the dataset's documented prompt reuse with different responses on each side, not a judged response the policy has seen. Those 72 prompts are nonetheless prompts the policy trained on, and [Section 6](#6-comparative-evaluation-protocol) draws its comparison prompts from the test split on the claim that no stage has seen them, so they are excluded from the comparison prompt pool.

The random preview surfaced one property that shaped the implementation. In one sampled pair (example 124449 in the EDA output) the two responses open with `Yes, that's righ...` and `Yeah, that's a v...`, sharing their first two characters, and the script's shared-opening panel prints a second (example 17111, `Okay then, ...` against `Oh okay.`). These are the cases in which TRL's default prompt extraction, the longest common character prefix, would cut the prompt inside a word. The explicit marker split of [Section 3.1](#31-the-trl-v1-application-programming-interface) exists because pairs like this are common in a corpus whose responses are conversational openers.

## 3. Implementation and Design Choices

### 3.1. The TRL v1 Application Programming Interface

`Anthropic/hh-rlhf` stores each pair as two full dialogues sharing a common prefix, which is TRL's 'implicit prompt' preference format, and `trl.DPOTrainer` will extract the prompt itself if none is supplied. That default is not used here. TRL's extraction takes the longest common *character* prefix of the two sides, which in HH-RLHF routinely runs past the final `\n\nAssistant:` marker into the shared opening words of the two responses and can end mid-word. The trainer then tokenises the prompt and the prompt-plus-completion separately and slices the completion off by prompt length, so a mid-word cut changes the byte-pair merges at the boundary and the sliced completion loses or corrupts its first tokens, which TRL reports as a tokenised-prompt mismatch warning on every affected pair. The pipeline therefore supplies an explicit `prompt` column, split at the marker with the PPO stage's extractor, so the prompt ends on the colon and each response keeps its leading space, a pre-tokenisation boundary at which the two tokenisations agree. This is also the prompt the length filter measures, so the prompt cap applies to the object the trainer sees. The trainer still appends EOS to both sides itself. One consequence of the v1 API matters for the design: its only length mechanism is `max_length`, which **truncates** (from the start or end per `truncation_mode`), and there is no filtering cap and no separate prompt cap. Truncation is not benign for preference pairs, because the two sides of an HH-RLHF pair differ mainly in the final assistant turn, so clipping tends to leave two near-identical prefixes whose preference label is uninformative. Length control is therefore implemented outside the trainer ([Section 3.2](#32-length-filtering-matching-the-earlier-stages)), and `max_length` is set only as a backstop that the pre-filter guarantees never binds.

### 3.2. Length Filtering Matching the Earlier Stages

`filter_pairs` drops, before the trainer sees them, every pair violating either cap. A pair survives only if its prompt (everything up to and including the final `\n\nAssistant:` marker, extracted with the PPO stage's own `extract_prompt`) fits `max_prompt_tokens` = 256, and both full dialogues, with EOS appended exactly as the trainer appends it, fit `max_pair_tokens` = 512. Both caps use filter semantics, never truncation, and both splits are filtered, the train split included.

The caps are not new numbers: 256 is the PPO stage's prompt cap and 512 is the RM stage's `max_length`, both fixed by those stages' EDA. Reproducing them here is a fairness requirement of the PPO-versus-DPO comparison, since it guarantees DPO trains on the same prompt distribution PPO optimised on and on the same length-admissible pairs the RM learnt from, so a quality difference between the two arms cannot be attributed to DPO having seen longer, shorter, or clipped data. The two caps overlap (a long pair usually has a long prompt), so the EDA's compound-filter table decomposes the joint retention into what each cap uniquely costs.

### 3.3. One Backbone for Policy and Reference

The policy is trained as a LoRA (low-rank adaptation, [Hu et al. (2022)](#ref-hu2022)) Parameter-Efficient Fine-Tuning (PEFT) model on the merged SFT backbone, and `ref_model=None` is passed to the trainer. As in the PPO stage, the trainer then recovers $\pi_{\text{ref}}$ by disabling the adapters, which is exact rather than approximate: a freshly initialised LoRA contributes $\Delta W = BA = 0$, so the adapter-disabled policy coincides with the merged SFT model, and the base stays frozen throughout. One copy of the Qwen2.5-0.5B backbone therefore serves both roles, halving the memory of the naive two-copy setup. During training, [`model_utils.CacheCleaner`](https://github.com/nhan-dam/rlhf-course/blob/main/src/common/model_utils.py) bounds PyTorch's reserved-memory pool, as in every stage. The SFT report's Section 5 presents the allocator analysis behind it.

### 3.4. Adapter Parity with the PPO Policy

The LoRA configuration is identical to the PPO policy adapter: rank 32, $\alpha = 64$, dropout 0.05, targets `q_proj` and `v_proj`. This is the second fairness requirement: the two arms of the comparison get the same trainable capacity attached to the same backbone, so neither can win by having more parameters to move. The $\alpha = 2r$ convention follows the earlier stages.

### 3.5. Learning Rate

The default is $10^{-4}$. The DPO paper's $5 \times 10^{-7}$, and the $10^{-7}$ to $10^{-6}$ range that is standard in the literature, are full-fine-tuning rates. This stage trains a LoRA adapter, whose update starts at zero and needs a larger step, which is why the earlier stages ran the same adapter shape at $10^{-4}$ (RM) and $10^{-5}$ (PPO). The value was chosen by the sweep in [Section 9.1](#91-learning-rate-sweep).

### 3.6. Beta, Loss Type, and Likelihood Displacement

$\beta$ prices drift from $\pi_{\text{ref}}$ inside the implicit reward, playing the role the KL coefficient plays in the PPO stage, but the two coefficients are not numerically comparable, so no attempt is made to match them. The default is 0.1, and the sweep over 0.05, 0.1 and 0.3 is in [Section 9.2](#92-beta-sweep). The loss defaults to the sigmoid form of [(2)](#eq-dpo-loss). A documented failure mode of that loss is **likelihood displacement**: the log-probabilities of chosen and rejected responses falling together, with the margin growing only because the rejected side falls faster. The monitoring signal is `logps/chosen` falling alongside `logps/rejected`, and the documented responses are switching `loss_type` to `'ipo'` ([Azar et al., 2024](#ref-azar2024)), whose bounded objective removes the incentive to push the margin without limit, or raising $\beta$. Both are configuration changes.

The signal is a matter of degree, not a binary. Relative to the reference, the chosen-side log-probability is lower in every completed run ([Table 5](#tab-dpo-runs)), so its sign cannot discriminate between configurations and only its magnitude can. The stage therefore selects its configuration on that magnitude ([Section 9.3](#93-selected-configuration)) rather than by switching the loss, and IPO remains untested here.

### 3.7. Configuration-Driven Experiments and Run Tracking

The stage uses the same machinery as the other three: `DPOTrainingConfig` is parsed with `transformers.HfArgumentParser` from CLI overrides or a JSON file, the run label is the hash of the full resolved configuration, and each run writes `config_<label>.json` and `metrics_<label>.json` to its own results directory, joined and ranked by the shared `aggregate_metrics.py` (DPO runs are listed by gate accuracy, a display order rather than the selection criterion of [Section 9.3](#93-selected-configuration)). The $\beta$ sweep is therefore three commands differing in one flag, each landing in its own directory.

Unlike the experimental PPO trainer, `DPOTrainer` is a standard `Trainer` subclass, so the two capabilities the PPO stage lacks return here: an interrupted run resumes from its latest checkpoint (the unchanged config hashes to the same label, so the checkpoint directory is found automatically), and `load_best_model_at_end` keeps the checkpoint with the lowest evaluation loss. The per-pair loss is monotone in that pair's margin, but the mean loss is not monotone in accuracy, so the two can pick different checkpoints. In five of the seven runs they do, and the subsample accuracy given up is at most 0.014, inside that subsample's 0.016 standard error. The gate records both measures.

### 3.8. Post-Training Gate

Training-time evaluation scores a seeded 1,000-pair subsample of the filtered test split every 500 steps, enough precision for checkpoint selection at bounded cost, and the best checkpoint is chosen on it. After training, the gate scores the **remaining** 6,033 filtered test pairs once, disjoint from the subsample so that the checkpoint is never selected on pairs the gate then scores, and persists the result: the implicit-reward pairwise accuracy (the fraction of held-out pairs where the implicit reward ranks chosen above rejected), the mean margin, the chosen and rejected log-probabilities, and the chosen and rejected implicit rewards, whose ratio to $\beta$ is the drift from the reference that [Section 3.6](#36-beta-loss-type-and-likelihood-displacement) selects on. The two roles mirror the RM stage's, with the one difference that DPO keeps them disjoint because it selects a checkpoint on the subsample where the RM only monitored on it. The expectation band is the same 0.6 to 0.7 that gated the RM. It works as a floor, not a selection criterion: within each sweep of [Section 9](#9-appendix-hyperparameter-sweeps), gate accuracy rises with drift from the reference, so the band can be reached by drifting further. Re-running a completed configuration recomputes exactly this block on the best checkpoint without retraining, via the same resume mechanism.

## 4. Training Configuration

The values below are the defaults. They are overridable from the command line or a JSON file (see [Section 3.7](#37-configuration-driven-experiments-and-run-tracking)).

| Hyperparameter | Value |
|---|---|
| Policy base | SFT model (merged), reference via adapter disabling |
| LoRA rank $r$ / scaling $\alpha$ / dropout | 32 / 64 / 0.05 (parity with the PPO policy) |
| LoRA target modules | `q_proj`, `v_proj` |
| $\beta$ | 0.1 (sweep axis: 0.05 / 0.1 / 0.3) |
| Loss type | sigmoid (IPO untested, see [Section 3.6](#36-beta-loss-type-and-likelihood-displacement)) |
| Learning rate | $10^{-4}$, linear schedule with 3% warmup |
| Epochs | 1 |
| Per-device batch size $\times$ gradient accumulation | $4 \times 4 = 16$ effective |
| Maximum prompt length | 256 tokens (filtering, not truncation) |
| Maximum pair length (each side, with EOS) | 512 tokens (filtering, not truncation) |
| Gradient checkpointing | disabled by default, configurable |
| Precision | bfloat16 |
| Evaluation | 1,000 filtered test pairs every 500 steps, best checkpoint kept, gate on the other 6,033 |

## 5. Training Diagnostics

- `rewards/accuracies`, the fraction of pairs whose implicit rewards are correctly ranked. It should climb from chance towards the 0.6 to 0.7 band. A flat curve at chance means the signal is too weak (raise the learning rate cautiously or check the data), while a rapid climb towards 1.0 suggests memorisation of the preference set rather than a generalisable ranking.
- `rewards/margins`, the mean implicit-reward margin, which should grow steadily. Margin growth with flat accuracy means existing correct pairs are being pushed further apart rather than new pairs being ranked correctly, which is the precursor of displacement.
- `logps/chosen` and `logps/rejected` together. Both falling is the likelihood-displacement signature of [Section 3.6](#36-beta-loss-type-and-likelihood-displacement). Its size, not its sign, decides the response, which is a configuration change either way.
- The evaluation loss, which selects the checkpoint. Divergence between falling train loss and rising evaluation loss is ordinary overfitting, expected within one epoch at this learning rate only if the rate is set too high.

## 6. Comparative Evaluation Protocol

The comparison against the PPO policy is designed before either arm is evaluated, so the protocol cannot drift towards whichever result looks better. It is implemented as a standing diagnostic, [`compare_policies.py`](https://github.com/nhan-dam/rlhf-course/blob/main/src/diagnostics/compare_policies.py), which takes the two run labels and writes `comparison_ppo_<ppo_label>_dpo_<dpo_label>_n<prompts>_k<samples>.json` (every per-prompt record, including each sample) and `.md` (summary table, head-to-head win rates with paired bootstrap intervals, and a reading sample) under a shared results directory. Every score is a per-prompt mean over four sampled completions before any comparison, the correction the PPO evaluation established, and the artefact name carries the prompt and sample counts so runs at different settings never overwrite one another.

Four policies are evaluated, not two. Alongside the PPO and DPO arms, the SFT model anchors how much preference optimisation added on top of instruction tuning, and the raw pre-SFT base model (Qwen2.5-0.5B) anchors how much the entire pipeline added on top of the pre-trained model. The anchors are reference points rather than arms: neither judge is calibrated on raw base-model text (the RM was initialised from the SFT model and trained on HH-RLHF dialogue), so the base row is read qualitatively, not by its scores.

- **Evaluation prompts from the dedicated test split.** Generation prompts are extracted from the 8,552-row test split (with the same 256-token prompt filter), which no stage of either pipeline has trained on. The 72 test prompts that HH-RLHF reuses from the training split with different responses ([Section 2.4](#24-data-quality-and-a-tokenisation-observation)) are excluded before sampling, so the claim holds at the prompt level and not only at the response level. The PPO stage's own 100 evaluation prompts were carved from the train split, which was sound for monitoring PPO in isolation but is unusable here, since those prompts sit inside DPO's training set.
- **Symmetric judging.** The RM is a biased judge (PPO was optimised against it directly), and DPO's implicit reward is biased in the mirror-image way. Both models' responses are therefore scored by both judges and reported as a 2$\times$2 table, with the RM stage's adversarial probes run against both models' outputs as a judge-independent check, and an external judge as tie-breaker if the two judges disagree.
- **The DPO arm is chosen on drift, never on RM score.** The sweeps produce several DPO models and one is compared against the PPO run. It is `75047d16`, selected in [Section 9.3](#93-selected-configuration) on gate accuracy against chosen-side drift. It is not chosen by RM score. A lower $\beta$ or a higher rate permits more drift from $\pi_{\text{ref}}$, more drift buys RM score even where the RM no longer tracks quality, and the RM cannot see drift. The comparison is arm against arm: each algorithm enters with its selected configuration, and the evidence for each selection stays in that algorithm's own report, here [Section 9](#9-appendix-hyperparameter-sweeps). Each policy's sampled KL from $\pi_{\text{ref}}$ is reported beside its scores so that drift stays visible.
- **Output diversity.** Distinct-bigram ratio and missing-EOS rate over the same generations (dependency-free stand-ins for self-BLEU), since preference optimisation can purchase margin with mode collapse or turn-closing failures.
- **Data budgets on record.** One DPO epoch sees every filtered training pair, whereas the PPO run consumed a 10,000-episode budget covering roughly 7.5% of its filtered prompts. The budgets cannot be meaningfully equalised, and are reported rather than hidden.

## 7. Results

The selected arm is `75047d16`, at a learning rate of $10^{-4}$ and $\beta = 0.1$. The reasons for choosing it over the six other runs are in [Section 9.3](#93-selected-configuration).

### 7.1. Training and Gate

<a id="tab-dpo-arm"></a>

| Measure | Value |
|---|---|
| Gate accuracy (6,033 pairs) | 0.635 |
| Gate margin / log-ratio gap h | 0.348 / 3.48 |
| Gate loss | 0.632 |
| Chosen / rejected drift from the reference | −6.44 / −9.92 nats |
| Subsample accuracy at step 500 / selected checkpoint | 0.556 / 0.643 |
| Training accuracy, last 200 steps | 0.636 |
| Selected checkpoint | step 7,500 of 8,308 |
| Gradient norm, median / maximum | 0.97 / 1.60 |

Table 3: The selected arm's gate figures, in-training figures, and optimisation statistics. Drift is the implicit reward divided by beta.

Against the expectations of [Section 5](#5-training-diagnostics), the run is good on every count but one.

- Accuracy climbed from 0.556 to 0.643 on the subsample and scored 0.635 on the gate, inside the band.
- There is no sign of memorisation. Training accuracy over the last 200 steps is 0.636, against 0.643 on the held-out subsample.
- Optimisation was stable throughout, with no gradient-norm excursion and evaluation runtime steady at 271 s.
- Likelihood displacement is present but mild. On the subsample, `logps/chosen` fell 1.35 nats from step 500 to the end while `logps/rejected` fell 4.04, so the margin was earned mainly by suppressing the rejected side. Relative to the reference, the chosen side sits 6.44 nats lower at the gate.

### 7.2. Generation Health

Gate accuracy ranks human-written pairs and says nothing about what the policy generates. [`probe_dpo_generations.py`](https://github.com/nhan-dam/rlhf-course/blob/main/src/diagnostics/probe_dpo_generations.py) samples both the arm and the SFT model on 40 held-out test prompts, twice each, at temperature 0.7 and up to 128 new tokens, and reports form statistics only. It computes no reward-model score and no win rate.

<a id="tab-dpo-probe"></a>

| Policy | Mean tokens | EOS rate | Hit 128-token cap | Distinct-4 | Log p per token | Implicit reward on own text |
|---|---|---|---|---|---|---|
| SFT | 92.9 | 0.53 | 38 of 80 | 0.978 | −1.251 | −0.658 |
| DPO arm | 80.1 | 0.72 | 22 of 80 | 0.973 | −1.327 | +0.654 |

Table 4: Form statistics over 80 completions per policy on paired prompts and seeds. The last column is the arm's implicit reward evaluated on each policy's own completions.

- Termination improved. Paired by prompt and sample, the arm emitted EOS where SFT did not on 24 completions, and the reverse happened on 8.
- The arm is shorter on average by 12.8 tokens, but the median paired difference is 0, so the gap comes from fewer completions running to the cap rather than from uniform shortening.
- Repetition is unchanged, at a distinct-4 ratio of 0.973 against 0.978.
- The implicit reward is not a neutral judge of generated text. The arm scores its own completions +0.654 and SFT's −0.658, a separation created by which policy wrote the text. This bears directly on the symmetric-judging design of [Section 6](#6-comparative-evaluation-protocol).

These statistics establish that the arm did not degenerate. They do not establish that its text is better than SFT's, which needs the blinded judging of [Section 8](#8-reflections-and-next-steps).

### 7.3. Comparison with PPO

Not yet run. The protocol is fixed in [Section 6](#6-comparative-evaluation-protocol).

## 8. Reflections and Next Steps

What the sweeps settled:

- **The published DPO learning rate does not transfer to a LoRA adapter.** It stayed near chance for a full epoch, and the band was cleared decisively only at a rate 200 times higher ([Section 9.1](#91-learning-rate-sweep)).
- **Raising $\beta$ holds the policy near the reference more efficiently than lowering the rate.** At $\beta = 0.3$ the policy matches the accuracy of the two lower rates at $\beta = 0.1$ with less than half their drift ([Section 9.3](#93-selected-configuration)). The two knobs are not interchangeable routes along one trade-off.
- **Gate accuracy cannot select a configuration on its own.** Within each sweep it rises with drift from the reference.
- **The displacement trigger needs a magnitude.** Its sign is the same in every run.

What remains open:

- **Generation quality.** [Section 7.2](#72-generation-health) shows the arm is not degenerate, not that it is better. The next measurement is a blinded pairwise judgement against SFT on 200 held-out prompts, stratified equally between the helpfulness and harmlessness subsets of HH-RLHF, with each subset judged under its own annotation question.
- **The comparison with PPO** of [Section 6](#6-comparative-evaluation-protocol). Two parts of its protocol have no tooling yet: the RM adversarial probes are fixtures rather than a pass over policy generations, and no external judge or prompt subset is chosen for the tie-breaker. Both must be settled before it runs.
- **Values of $\beta$ between 0.1 and 0.3.** No run was trained there, so that stretch of the accuracy–drift frontier is unmeasured.
- **IPO** is untested. It becomes worth running if generation quality shows a cost that a higher $\beta$ cannot recover.

## 9. Appendix: Hyperparameter Sweeps

Seven runs were trained, each for one epoch of 8,308 steps with a linear schedule decaying to zero after 249 warmup steps, and the seed 42. Each differs from the arm `75047d16` in exactly one field, the learning rate or $\beta$. Labels are hashes of the resolved configuration, and each run's `config_<label>.json` is the authoritative record of what produced it.

Two evaluation populations appear below and are not interchangeable. The in-training evaluation scores a 1,000-pair subsample every 500 steps and drives checkpoint selection, with a standard error near 0.016 on accuracy. The post-training gate scores the disjoint 6,033-pair remainder once, on the selected checkpoint, with a standard error near 0.006. Runs are compared on the gate. Because TRL logs the implicit reward scaled by $\beta$, two derived quantities are used wherever $\beta$ differs: the log-ratio gap $h$, i.e. the margin divided by $\beta$, and the drift of each side, i.e. its implicit reward divided by $\beta$, in nats relative to the reference policy.

<a id="tab-dpo-runs"></a>

| Run | Learning rate | Beta | Gate accuracy | Gate margin | h | Chosen drift | Rejected drift | Gate loss | Selected checkpoint | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `acb632d8` | $5 \times 10^{-7}$ | 0.1 | 0.553 | 0.016 | 0.16 | −0.63 | −0.80 | 0.687 | 7,000 | Bad |
| `0ae253f0` | $10^{-5}$ | 0.1 | 0.587 | 0.107 | 1.07 | −3.47 | −4.54 | 0.669 | 8,000 | Bad |
| `38139929` | $3 \times 10^{-5}$ | 0.1 | 0.601 | 0.196 | 1.96 | −4.84 | −6.80 | 0.655 | 8,308 | Bad |
| `75047d16` | $10^{-4}$ | 0.1 | 0.635 | 0.348 | 3.48 | −6.44 | −9.92 | 0.632 | 7,500 | Good, the arm |
| `c5d8a287` | $3 \times 10^{-4}$ | 0.1 | 0.642 | 0.483 | 4.83 | −12.90 | −17.73 | 0.639 | 8,308 | Mixed |
| `65674cc2` | $10^{-4}$ | 0.05 | 0.649 | 0.382 | 7.64 | −18.76 | −26.40 | 0.616 | 8,308 | Mixed |
| `4e27ae3a` | $10^{-4}$ | 0.3 | 0.603 | 0.295 | 0.98 | −1.50 | −2.49 | 0.667 | 7,000 | Mixed |

Table 5: All seven runs, scored on the 6,033-pair gate at each run's selected checkpoint. The first five rows are the learning-rate sweep at beta 0.1, and the last two complete the beta sweep at a learning rate of 1e-4. h is the gate margin divided by beta, and drift is each side's gate implicit reward divided by beta, in nats relative to the reference policy.

Every run trained stably, with no gradient-norm excursion, and none memorised its training pairs: in every run, training accuracy over the last 200 steps is within 0.021 of the final subsample accuracy.

### 9.1. Learning-Rate Sweep

The rate was swept over five values at $\beta = 0.1$, the first five rows of [Table 5](#tab-dpo-runs). Gate accuracy and chosen-side drift both rise monotonically with the rate.

- $5 \times 10^{-7}$, the DPO paper's full-fine-tuning rate, is a near-null on a LoRA adapter. After a full epoch its gate margin is 0.016 and its accuracy 0.553. Bad.
- $10^{-5}$ and $3 \times 10^{-5}$ reach 0.587 and 0.601, below and at the floor of the 0.6 to 0.7 band. Bad.
- $10^{-4}$ is the lowest rate to clear the band with room to spare. It is the arm.
- $3 \times 10^{-4}$ gains 0.007 over the arm, about one gate standard error, for twice the chosen-side drift. Its gate loss is also higher, at 0.639 against 0.632, so loss and accuracy rank these two runs in opposite orders. Mixed.

One property of the sweep bears on how any future rate trial should be read. Accuracy lags the margin: at step 500 the $3 \times 10^{-5}$ and $10^{-4}$ runs score 0.551 and 0.556 on the subsample, a third of a standard error apart, while their margins are already 0.065 and 0.108. Their accuracy curves separate only by step 1,500, at 0.559 and 0.593. A rate trial judged on accuracy that early would have discarded the rate that worked.

### 9.2. Beta Sweep

$\beta$ was swept over 0.05, 0.1 and 0.3 at a learning rate of $10^{-4}$, i.e. the arm and the last two rows of [Table 5](#tab-dpo-runs). Lowering $\beta$ raises accuracy and drift together: from $\beta = 0.3$ to 0.05, gate accuracy rises from 0.603 to 0.649 while chosen-side drift grows from −1.50 to −18.76 nats. The rejected side drifts further than the chosen side in all three, so each margin is earned mainly by suppressing the rejected response. The median gradient norm rises with $\beta$, from 0.66 at 0.05 to 2.29 at 0.3, as expected from the factor of $\beta$ in the DPO gradient.

The subsample and the gate disagree on $\beta = 0.3$. On the subsample at step 8,308 it scores 0.643, level with the arm, whereas the gate places it 0.032 below. Two differences account for it: the subsample's standard error is 0.016 against the gate's 0.006, and its final row is the last checkpoint whereas the gate scores the selected one, step 7,000 for $\beta = 0.3$. This is why runs are compared on the gate.

### 9.3. Selected Configuration

The two sweeps together show that the knobs are not interchangeable. Within each sweep, accuracy rises with drift. Across them, $\beta = 0.3$ reaches 0.603 at −1.50 nats, matching or exceeding $10^{-5}$ and $3 \times 10^{-5}$ at $\beta = 0.1$ while drifting less than half as far as either. Those two rates are therefore dominated, i.e. another run has both higher accuracy and less drift. The remaining five runs form the accuracy–drift frontier: $5 \times 10^{-7}$, $\beta = 0.3$, the arm, $3 \times 10^{-4}$ and $\beta = 0.05$, in order of increasing drift. The verdicts in [Table 5](#tab-dpo-runs) follow from this: Bad marks a run below the band or dominated, Mixed a frontier run that trades one axis for the other against the arm, and Good the arm.

One rule decides every run: a configuration displaces the arm only by scoring higher on the gate at chosen-side drift no worse than the arm's −6.44 nats. No run meets it. Every run with less drift scores lower, and the two that score higher drift two and three times as far.

- `65674cc2` at $\beta = 0.05$ gains 0.014 accuracy for 2.9 times the drift.
- `c5d8a287` at $3 \times 10^{-4}$ gains 0.007 for 2.0 times the drift.
- `4e27ae3a` at $\beta = 0.3$ cuts drift to 0.23 times the arm's but loses 0.032 accuracy, the only gap of the three well beyond the gate's standard error.

The arm is therefore `75047d16`, carried in `configs/dpo_default.json` and promoted to `dpo-model/`. The selection rests on the gate and the training diagnostics. The arm's generations have been checked for form ([Section 7.2](#72-generation-health)) but not yet judged for quality.

## 10. References

- <span id="ref-azar2024"></span>Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Rémi Munos, Mark Rowland, Michal Valko, Daniele Calandriello. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS 2024. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036).
- <span id="ref-hu2022"></span>Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- <span id="ref-rafailov2023"></span>Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn. *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.* NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
