# Direct Preference Optimisation Fine-Tuning on Anthropic HH-RLHF

> Created on: 17 July 2026
>
> Updated on: 22 September 2026

This note documents an implementation of the first Phase 3 stage: direct preference optimisation (DPO) ([Rafailov et al., 2023](#ref-rafailov2023)) of the supervised fine-tuning (SFT) policy on the pairwise preferences of `Anthropic/hh-rlhf`, replacing the reward model (RM) and proximal policy optimisation (PPO) stages of the classical reinforcement learning from human feedback (RLHF) pipeline with a single supervised loss. The stage is designed as the second arm of a controlled PPO-versus-DPO comparison: its data view, trainable capacity, and initialisation deliberately match the PPO stage documented in the PPO report.

The full source code can be found on [GitHub](https://github.com/nhan-dam/rlhf-course/blob/main/src/pipeline/dpo_lora_hh.py).

**Status note.** Implementation, exploratory data analysis and both hyperparameter sweeps are complete. The selected arm is `75047d16`, at a learning rate of $10^{-4}$ and $\beta = 0.1$. Its results are in [Section 7](#7-results) and every run is in [Section 9](#9-appendix-hyperparameter-sweeps). The comparison against PPO in [Section 6](#6-comparative-evaluation-protocol) has not yet been run.

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

The default is $10^{-4}$. The DPO paper's $5 \times 10^{-7}$, and the $10^{-7}$ to $10^{-6}$ range that is standard in the literature, are full-fine-tuning rates. This stage trains a LoRA adapter, whose update starts at zero and needs a larger step, which is why the earlier stages ran the same adapter shape at $10^{-4}$ (RM) and $10^{-5}$ (PPO). The value was set by the sweep in [Section 9.1](#91-learning-rate-sweep). Of five rates, the three below $10^{-4}$ missed the gate band, and $3 \times 10^{-4}$ cleared it by a further 0.007 at twice the chosen-side drift. A higher rate buys accuracy with drift in the same way a lower $\beta$ does ([Table 7](#tab-dpo-completed-runs)).

### 3.6. Beta, Loss Type, and Likelihood Displacement

$\beta$ prices drift from $\pi_{\text{ref}}$ inside the implicit reward, playing the role the KL coefficient plays in the PPO stage, but the two coefficients are not numerically comparable, so no attempt is made to match them. The default is 0.1, and the sweep over 0.05, 0.1 and 0.3 is in [Section 9.2](#92-beta-sweep). The loss defaults to the sigmoid form of [(2)](#eq-dpo-loss). A documented failure mode of that loss is **likelihood displacement**: the log-probabilities of chosen and rejected responses falling together, with the margin growing only because the rejected side falls faster. The monitoring signal is `logps/chosen` falling alongside `logps/rejected`, and the documented responses are switching `loss_type` to `'ipo'` ([Azar et al., 2024](#ref-azar2024)), whose bounded objective removes the incentive to push the margin without limit, or raising $\beta$. Both are configuration changes.

The signal is a matter of degree, not a binary. Relative to the reference, the chosen-side log-probability is lower in every completed run, so its sign cannot discriminate between configurations. Its magnitude can: at the gate it ranges from −1.50 nats at $\beta = 0.3$ to −18.76 at $\beta = 0.05$ ([Table 7](#tab-dpo-completed-runs)). The stage therefore selects $\beta$ on that magnitude ([Section 9.3](#93-selected-configuration)) rather than switching the loss, and IPO remains untested here.

### 3.7. Configuration-Driven Experiments and Run Tracking

The stage uses the same machinery as the other three: `DPOTrainingConfig` is parsed with `transformers.HfArgumentParser` from CLI overrides or a JSON file, the run label is the hash of the full resolved configuration, and each run writes `config_<label>.json` and `metrics_<label>.json` to its own results directory, joined and ranked by the shared `aggregate_metrics.py` (DPO runs rank by held-out implicit-reward accuracy). The $\beta$ sweep is therefore three commands differing in one flag, each landing in its own directory.

Unlike the experimental PPO trainer, `DPOTrainer` is a standard `Trainer` subclass, so the two capabilities the PPO stage lacks return here: an interrupted run resumes from its latest checkpoint (the unchanged config hashes to the same label, so the checkpoint directory is found automatically), and `load_best_model_at_end` keeps the checkpoint with the lowest evaluation loss. The per-pair loss is monotone in that pair's margin, but the mean loss is not monotone in accuracy. Across runs, `c5d8a287` has both the higher gate accuracy and the higher gate loss of the pair it forms with `75047d16`, at 0.642 against 0.635 and 0.639 against 0.632. Within a run, the lowest-loss checkpoint is also the highest-accuracy one in two of the four completed runs, and in the other two the subsample accuracy given up is 0.001 and 0.006, inside its 0.016 standard error.

### 3.8. Post-Training Gate

Training-time evaluation scores a seeded 1,000-pair subsample of the filtered test split every 500 steps, enough precision for checkpoint selection at bounded cost, and the best checkpoint is chosen on it. After training, the gate scores the **remaining** 6,033 filtered test pairs once, disjoint from the subsample so that the checkpoint is never selected on pairs the gate then scores, and persists the result: the implicit-reward pairwise accuracy (the fraction of held-out pairs where the implicit reward ranks chosen above rejected), the mean margin, the chosen and rejected log-probabilities, and the chosen and rejected implicit rewards, whose ratio to $\beta$ is the drift from the reference that [Section 3.6](#36-beta-loss-type-and-likelihood-displacement) selects on. The two roles mirror the RM stage's, with the one difference that DPO keeps them disjoint because it selects a checkpoint on the subsample where the RM only monitored on it. The expectation band is the same 0.6 to 0.7 that gated the RM. All four completed runs land inside it, between 0.603 and 0.649, and [Table 7](#tab-dpo-completed-runs) shows why reaching it says little on its own: across configurations, gate accuracy rises with drift from the reference. The band is therefore a floor, not a selection criterion. Re-running a completed configuration recomputes exactly this block on the best checkpoint without retraining, via the same resume mechanism.

## 4. Training Configuration

The values below are the defaults. They are overridable from the command line or a JSON file (see [Section 3.7](#37-configuration-driven-experiments-and-run-tracking)).

| Hyperparameter | Value |
|---|---|
| Policy base | SFT model (merged), reference via adapter disabling |
| LoRA rank $r$ / scaling $\alpha$ / dropout | 32 / 64 / 0.05 (parity with the PPO policy) |
| LoRA target modules | `q_proj`, `v_proj` |
| $\beta$ | 0.1 (sweep axis: 0.05 / 0.1 / 0.3) |
| Loss type | sigmoid (switch to IPO on likelihood displacement) |
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
- `logps/chosen` and `logps/rejected` together. Both falling is the likelihood-displacement signature of [Section 3.6](#36-beta-loss-type-and-likelihood-displacement), and the response (IPO or higher $\beta$) is a configuration change.
- The evaluation loss, which selects the checkpoint. Divergence between falling train loss and rising evaluation loss is ordinary overfitting, expected within one epoch at this learning rate only if the rate is set too high.

## 6. Comparative Evaluation Protocol

The comparison against the PPO policy is designed before either arm is evaluated, so the protocol cannot drift towards whichever result looks better. It is implemented as a standing diagnostic, [`compare_policies.py`](https://github.com/nhan-dam/rlhf-course/blob/main/src/diagnostics/compare_policies.py), which takes the two run labels and writes `comparison_ppo_<ppo_label>_dpo_<dpo_label>_n<prompts>_k<samples>.json` (every per-prompt record, including each sample) and `.md` (summary table, head-to-head win rates with paired bootstrap intervals, and a reading sample) under a shared results directory. Every score is a per-prompt mean over four sampled completions before any comparison, the correction the PPO evaluation established, and the artefact name carries the prompt and sample counts so runs at different settings never overwrite one another.

Four policies are evaluated, not two. Alongside the PPO and DPO arms, the SFT model anchors how much preference optimisation added on top of instruction tuning, and the raw pre-SFT base model (Qwen2.5-0.5B) anchors how much the entire pipeline added on top of the pre-trained model. The anchors are reference points rather than arms: neither judge is calibrated on raw base-model text (the RM was initialised from the SFT model and trained on HH-RLHF dialogue), so the base row is read qualitatively, not by its scores.

- **Evaluation prompts from the dedicated test split.** Generation prompts are extracted from the 8,552-row test split (with the same 256-token prompt filter), which no stage of either pipeline has trained on. The 72 test prompts that HH-RLHF reuses from the training split with different responses ([Section 2.4](#24-data-quality-and-a-tokenisation-observation)) are excluded before sampling, so the claim holds at the prompt level and not only at the response level. The PPO stage's own 100 evaluation prompts were carved from the train split, which was sound for monitoring PPO in isolation but is unusable here, since those prompts sit inside DPO's training set.
- **Symmetric judging.** The RM is a biased judge (PPO was optimised against it directly), and DPO's implicit reward is biased in the mirror-image way. Both models' responses are therefore scored by both judges and reported as a 2$\times$2 table, with the RM stage's adversarial probes run against both models' outputs as a judge-independent check, and an external judge as tie-breaker if the two judges disagree.
- **The DPO arm is chosen on drift, never on RM score.** The sweeps produce several DPO models and one is compared against the PPO run. It is `75047d16`, selected in [Section 9.3](#93-selected-configuration) on gate accuracy against chosen-side drift. It is not chosen by RM score. A lower $\beta$ or a higher rate permits more drift from $\pi_{\text{ref}}$, more drift buys RM score even where the RM no longer tracks quality, and the RM cannot see drift. The other completed runs are reported beside the comparison as a sensitivity check. Each policy's sampled KL from $\pi_{\text{ref}}$ is reported beside its scores so that drift stays visible.
- **Output diversity.** Distinct-bigram ratio and missing-EOS rate over the same generations (dependency-free stand-ins for self-BLEU), since preference optimisation can purchase margin with mode collapse or turn-closing failures.
- **Data budgets on record.** One DPO epoch sees every filtered training pair, whereas the PPO run consumed a 10,000-episode budget covering roughly 7.5% of its filtered prompts. The budgets cannot be meaningfully equalised, and are reported rather than hidden.

## 7. Results

The selected arm is `75047d16`, at a learning rate of $10^{-4}$ and $\beta = 0.1$. The reasons for choosing it over the other completed runs are in [Section 9.3](#93-selected-configuration).

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

- **The learning rate is bounded on both sides by measurement.** Below $10^{-4}$ no rate reached the band. Above it, $3 \times 10^{-4}$ bought 0.007 accuracy for twice the drift.
- **Gate accuracy cannot select a configuration on its own.** Across all four completed runs it rises with drift from the reference, whichever knob produced the drift ([Table 7](#tab-dpo-completed-runs)).
- **The displacement trigger needs a magnitude.** Its sign is the same in every completed run, while its size ranges from −1.50 to −18.76 nats.

What remains open:

- **Generation quality.** [Section 7.2](#72-generation-health) shows the arm is not degenerate, not that it is better. A blinded pairwise judgement against SFT, with the two HH-RLHF question texts applied to their own subsets, is the next measurement.
- **The comparison with PPO** of [Section 6](#6-comparative-evaluation-protocol). Two parts of its protocol have no tooling yet: the RM adversarial probes are fixtures rather than a pass over policy generations, and no external judge or prompt subset is chosen for the tie-breaker. Both must be settled before it runs.
- **IPO** is untested. It becomes worth running if generation quality shows a cost that a higher $\beta$ cannot recover without the accuracy loss seen at 0.3.

## 9. Appendix: Hyperparameter Sweeps

Seven runs were trained in total. Every run from `38139929` onwards differs from the arm `75047d16` in exactly one field, the learning rate or $\beta$. The two earliest, `e7492fba` and `7bc05fb0`, also predate the warmup field and ran without warmup. All use one epoch of 8,308 steps, a linear schedule decaying to zero, and the seed 42. Labels are hashes of the resolved configuration, so a run's `config_<label>.json` is the authoritative record of what produced it. Fields added to the configuration since the earliest runs mean the same settings would hash differently today.

Two evaluation populations appear below and are not interchangeable. The in-training evaluation scores a 1,000-pair subsample every 500 steps and drives checkpoint selection, giving a standard error near 0.016 on accuracy. The post-training gate scores the disjoint 6,033-pair remainder once, on the checkpoint selected by evaluation loss, giving a standard error near 0.006. Gate figures are the comparable ones across runs, since every run reaches the gate by the same procedure.

### 9.1. Learning-Rate Sweep

The rate was swept at $\beta = 0.1$ throughout, over five values. The published DPO rate of $5 \times 10^{-7}$ and the $10^{-7}$ to $10^{-6}$ range in the surrounding literature are full-fine-tuning rates. A LoRA adapter starts at zero, so they were treated as a starting point rather than a default.

<a id="tab-dpo-lr-sweep"></a>

| Run | Learning rate | Steps | Accuracy at 500 / 1000 / 1500 | Margin at 500 / 1000 / 1500 | Chosen drift at 1500 | Outcome |
|---|---|---|---|---|---|---|
| `e7492fba` | $5 \times 10^{-7}$ | 2,500 of 8,308 | 0.510 / 0.504 / 0.528 | 0.003 / 0.003 / 0.007 | −0.17 | Bad, stopped |
| `7bc05fb0` | $10^{-5}$ | 1,500 of 8,308 | 0.548 / 0.543 / 0.550 | 0.038 / 0.058 / 0.071 | −2.07 | Bad, stopped |
| `38139929` | $3 \times 10^{-5}$ | 1,500 of 8,308 | 0.551 / 0.562 / 0.559 | 0.065 / 0.085 / 0.100 | −2.63 | Bad, stopped |
| `75047d16` | $10^{-4}$ | 8,308 of 8,308 | 0.556 / 0.559 / 0.593 | 0.108 / 0.128 / 0.178 | −4.21 | Good, gate 0.635 |
| `c5d8a287` | $3 \times 10^{-4}$ | 8,308 of 8,308 | 0.568 / 0.570 / 0.610 | 0.143 / 0.173 / 0.301 | −11.85 | Mixed, gate 0.642 |

Table 5: The five learning-rate runs at beta = 0.1, with in-training accuracy and implicit-reward margin at the first three evaluations, the chosen-side drift at step 1,500 in nats relative to the reference policy, i.e. the chosen reward divided by beta, and the outcome of each.

The three lowest rates are unambiguous failures. At $5 \times 10^{-7}$ the margin moved 0.007 in 1,500 steps and accuracy stayed within one standard error of chance, a null result. At $10^{-5}$ and $3 \times 10^{-5}$ accuracy reached 0.550 and 0.559 by step 1,500, both inside one standard error of each other and far below the 0.6 to 0.7 band. The two highest rates both completed and both cleared the band, at 0.635 and 0.642.

$3 \times 10^{-4}$ was run last, after the beta sweep, to test whether the rate had been left too low. It trained cleanly, with a median gradient norm of 1.17 against 0.97 at $10^{-4}$, a maximum of 2.06 against 1.60, and no instability. It also selected its final checkpoint rather than an earlier one, so it did not overfit by evaluation loss within the epoch. Its gate accuracy is 0.642 against 0.635, a gap of about one gate standard error, while its chosen-side drift is −12.90 nats against −6.44, and its gate loss is worse at 0.639 against 0.632. Over half of its total drift was already present at its first evaluation, which recorded −9.67 nats at step 500 against −4.74 for $10^{-4}$.

Two properties of this sweep are worth recording, because both were initially misread.

- Accuracy lags the margin. At step 500 the $3 \times 10^{-5}$ and $10^{-4}$ runs stand at 0.551 and 0.556, a third of a standard error apart, while their margins are 0.065 and 0.108. The accuracy curves separate only at step 1,500. A learning-rate trial read on accuracy before step 2,000 would have discarded the rate that worked.
- The rate and likelihood displacement move together. The drift column is monotone in the rate, from −0.17 nats at $5 \times 10^{-7}$ to −11.85 at $3 \times 10^{-4}$, so the accuracy the higher rates gained was bought with movement away from the reference policy. Displacement was not what held the three stopped runs back, and the rate was raised rather than $\beta$ or the loss type changed. The same drift measure is the axis on which $\beta$ is chosen in [Section 9.2](#92-beta-sweep), so the two sweeps are read against one criterion.

### 9.2. Beta Sweep

With the rate fixed at $10^{-4}$, $\beta$ was swept over 0.05, 0.1 and 0.3, each for a full epoch. All three reached the gate. Because TRL scales the logged reward by $\beta$, the raw margin is not comparable across these runs, so the table also reports the underlying log-ratio gap $h$, i.e. the margin divided by $\beta$, and the chosen and rejected drift in nats relative to the reference policy.

<a id="tab-dpo-beta-sweep"></a>

| Run | Beta | Gate accuracy | Gate margin | Gate h | Gate loss | Chosen drift | Rejected drift | Median grad norm | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `65674cc2` | 0.05 | 0.649 | 0.382 | 7.64 | 0.616 | −18.76 | −26.40 | 0.66 | Mixed |
| `75047d16` | 0.1 | 0.635 | 0.348 | 3.48 | 0.632 | −6.44 | −9.92 | 0.97 | Good |
| `4e27ae3a` | 0.3 | 0.603 | 0.295 | 0.98 | 0.667 | −1.50 | −2.49 | 2.29 | Mixed |

Table 6: The three beta runs at a learning rate of 1e-4, scored on the 6,033-pair gate at the checkpoint selected by evaluation loss. Drift columns are the chosen-side and rejected-side rewards divided by beta, in nats relative to the reference policy.

All three runs completed without instability. Evaluation runtime held between 271 and 277 s, evaluation loss fell monotonically in each, and no run showed a gradient-norm excursion, although the median gradient norm rises with $\beta$ as expected from the factor of $\beta$ in the gradient.

The three sit on a single monotone trade-off, and neither end of it is free.

- $\beta = 0.05$ has the best gate accuracy at 0.649 and the worst drift by a wide margin, with the policy 18.76 nats below the reference on chosen responses. It gains 0.014 accuracy over $\beta = 0.1$, which is close to twice the gate standard error, in exchange for roughly three times the chosen-side drift. Mixed.
- $\beta = 0.1$ sits in the middle on every column. Good.
- $\beta = 0.3$ has the least drift at 1.50 nats and the worst gate accuracy at 0.603, 0.032 below $\beta = 0.1$, which is about five gate standard errors. Mixed.

One measurement caution follows from the two populations. On the 1,000-pair in-training subsample at step 8,308 the three runs score 0.656, 0.641 and 0.643 for $\beta$ of 0.05, 0.1 and 0.3, which would place $\beta = 0.3$ level with $\beta = 0.1$. The gate reverses that. Two differences account for it: the subsample carries a standard error near 0.016 against the gate's 0.006, and the subsample row is the final checkpoint whereas the gate scores the selected checkpoint, which is step 7,000 for $\beta = 0.3$, 7,500 for $\beta = 0.1$ and 8,308 for $\beta = 0.05$. Comparisons between runs are drawn from the gate for this reason.

### 9.3. Selected Configuration

Four runs completed a full epoch and reached the gate. Sorted by chosen-side drift, they order identically on accuracy, and the two axes vary together whichever knob produced the drift.

<a id="tab-dpo-completed-runs"></a>

| Run | Learning rate | Beta | Gate accuracy | Chosen drift |
|---|---|---|---|---|
| `4e27ae3a` | $10^{-4}$ | 0.3 | 0.603 | −1.50 |
| `75047d16` | $10^{-4}$ | 0.1 | 0.635 | −6.44 |
| `c5d8a287` | $3 \times 10^{-4}$ | 0.1 | 0.642 | −12.90 |
| `65674cc2` | $10^{-4}$ | 0.05 | 0.649 | −18.76 |

Table 7: The four completed runs ordered by chosen-side drift, with gate accuracy on the 6,033 pairs and drift in nats relative to the reference policy, i.e. the chosen reward divided by beta.

Two knobs produced this spread. Rows two to four hold $\beta$ at 0.1 or vary it at a fixed rate, and the ordering does not distinguish them: the run that drifted furthest scored highest whether the drift came from a lower $\beta$ or a higher rate. Gate accuracy on this stage is therefore not independent of how far the policy moved from the reference, which is what makes it unusable as the sole selection criterion.

The main-text arm is `75047d16`, i.e. a learning rate of $10^{-4}$ and $\beta = 0.1$, with the sigmoid loss and the adapter of [Section 3.4](#34-adapter-parity-with-the-ppo-policy). It is carried in `configs/dpo_default.json`.

Three rates below $10^{-4}$ failed to clear the band at all ([Table 5](#tab-dpo-lr-sweep)), which fixes the lower bound. Above it, the remaining three completed runs each buy accuracy with drift, and the same rule was applied to all three: a configuration displaces the arm only by scoring higher at drift no worse than the arm's.

- `65674cc2` at $\beta = 0.05$ gains 0.014 accuracy for 2.9 times the drift.
- `c5d8a287` at $3 \times 10^{-4}$ gains 0.007, about one gate standard error, for 2.0 times the drift, and its gate loss is worse at 0.639 against 0.632.
- `4e27ae3a` at $\beta = 0.3$ cuts drift to 0.23 times the arm's but loses 0.032 accuracy, the largest gap in the table.

None meets the rule, so the arm stands. Likelihood displacement is the failure mode this stage was built to watch ([Section 3.6](#36-beta-loss-type-and-likelihood-displacement)), and at $\beta = 0.1$ and $10^{-4}$ the chosen-side log-probability falls 1.35 nats across the epoch while the rejected side falls 4.04, so the margin is earned mainly by suppressing the rejected side. At $\beta = 0.05$ the corresponding falls are 10.33 and 17.06.

This selection rests on the gate and the training diagnostics. The arm's generations have been checked for form ([Section 7.2](#72-generation-health)) but not judged for quality, and the comparison of [Section 6](#6-comparative-evaluation-protocol) has not been run.

## 10. References

- <span id="ref-azar2024"></span>Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Rémi Munos, Mark Rowland, Michal Valko, Daniele Calandriello. *A General Theoretical Paradigm to Understand Learning from Human Preferences.* AISTATS 2024. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036).
- <span id="ref-hu2022"></span>Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- <span id="ref-rafailov2023"></span>Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, Chelsea Finn. *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.* NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
