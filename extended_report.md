# Dual System VLA on LIBERO-Long: Extended Report

Design, evaluation methodology, results, and discussion in full.

This is the long form of the three page report. The short version states the
findings; this one states how each finding was produced, which alternatives were
rejected on the way, and how far each number can be pushed before it stops being
supported. Every table here is reproduced from the artifacts listed in Appendix A,
so any number can be recomputed from the shipped record.

**Reading order.** Sections 1 and 2 set up the questions. Sections 3 and 4 are
design and implementation. Section 5 is the evaluation protocol, which is where
the study either is or is not sound. Section 6 is results, Section 7 the answers,
Sections 8 to 10 the caveats and what comes next.

**Conventions.** Each experimental variation is a "condition". Success rates are
pooled proportions over paired episodes. Intervals are 95 percent Wilson score
intervals. Paired significance is McNemar's exact two sided test on discordant
episodes.

---

## Contents

1. [Introduction](#1-introduction)
2. [Problem statement](#2-problem-statement)
3. [System design](#3-system-design)
4. [Implementation](#4-implementation)
5. [Evaluation design](#5-evaluation-design)
6. [Results](#6-results)
7. [Discussion: answering the questions](#7-discussion-answering-the-questions)
8. [Limitations](#8-limitations)
9. [Lessons learned](#9-lessons-learned)
10. [Future work](#10-future-work)
- [Appendix A: where every number comes from](#appendix-a-where-every-number-comes-from)
- [Appendix B: condition name map](#appendix-b-condition-name-map)
- [Appendix C: reproduction commands](#appendix-c-reproduction-commands)
- [Appendix D: corrections against the compact report](#appendix-d-corrections-against-the-compact-report)

---

## 1. Introduction

A Dual System Vision Language Action (VLA) policy splits control into two modules
that run at different rates. System 2 is slow and semantic: it reasons about the
task, the scene, and where the robot currently is in a long horizon plan. System 1
is fast and reactive: it produces motor commands at control frequency, conditioned
on whatever System 2 last decided.

The motivation is specific rather than general. On long horizon manipulation, a
single system policy usually fails not because it cannot produce a good reaching
motion, but because it loses track of which stage of a multi stage task it is in.
It grasps an object it has already placed, or it moves to close a drawer that is
still empty. A module whose only job is "where am I in the plan" should remove that
failure mode, and the rest of the policy can stay a competent short horizon
controller.

This work builds such a policy on the LIBERO-Long (`libero_10`) benchmark, ships it
as an out of tree LeRobot plugin rather than a fork, and subjects it to a seven
condition ablation designed so that each condition isolates one claim.

**The principal result is negative, and the negative result is more informative
than a positive one would have been.** The conditioning interface demonstrably
works: corrupting the sub-goal signal costs 16.7 percentage points of task success
(50.7 percent against 34.0 percent, paired p below 1e-7), which establishes that
System 1 reads and acts on the channel. Given that, neither online selection by a
learned System 2, nor sub-goals taken from a ground truth oracle, improves success
over a fixed open loop schedule. On this System 1, the architecture is execution
bound, and the reasoning half has no headroom left to exploit.

The shipped system runs on a single 25-skill vocabulary (`v25`); no vocabulary
comparison is made in this report. Two secondary results are reported here that the
compact report has no room for:

* **The training budget result.** Success was still climbing at the 100,000 step
  budget where the shipped campaign stops (40.0, 45.0, 51.0 percent at 40k, 70k and
  100k on one seed), so the absolute numbers are not a ceiling (Section 6.7).
* **A second campaign at 200,000 steps**, which was measured but not folded into the
  compact report, and which reverses the direction of that budget trend for the
  conditioned condition while leaving the floor flat (Section 6.8).

---

## 2. Problem statement

LIBERO-Long contains ten long horizon manipulation tasks, each a composition of two
to four stages. A representative instruction is "put the black bowl in the bottom
drawer of the cabinet and close it", which decomposes into grasp the bowl, place the
bowl in the drawer, close the drawer. Episodes run up to 520 simulator steps at
10 fps. A single system policy must track progress through those stages implicitly,
from pixels and proprioception alone.

The exercise weights the ablation study rather than the raw success rate, so the
questions below are the deliverable and the success rate is a means to answering
them. They are numbered here and every later section refers back to these numbers.

### Q1. Counterfactual: does System 2 contribute anything?

**Q1.1 Does conditioning matter at all?**
If the sub-goal channel is removed entirely, does success fall? If it does not, the
second system is decoration and the architecture is unjustified. The question has a
second, sharper form: if the channel is deliberately corrupted rather than removed,
does success fall? That version is a positive control on the whole instrument.

**Q1.2 Does *online* selection matter, or only the decomposition?**
A dual system policy makes two separate claims: that decomposing a task into
sub-goals helps, and that choosing between them at control frequency, in response to
what is happening, helps. These are separable. A fixed time based schedule keeps the
decomposition and discards the closed loop, and comparing against it isolates the
second claim from the first.

**Q1.3 Is the naive static baseline sufficient?**
The brief names a specific naive alternative: one fixed instruction embedding for
the whole episode. If that performs as well as a per step selector, the complexity
of System 2 is not paying for itself.

### Q2. Error recovery: does System 2 help under perturbation?

**Q2.1 Does the system detect a disturbance?**
When the task is knocked backwards, for example by forcing the gripper open while it
is carrying an object, does System 2 notice and revise its belief about where the
robot is in the plan? This is a question about detection, and it is measurable
independently of whether the robot then succeeds.

**Q2.2 Does detection translate into recovery?**
Detecting a disturbance is worthwhile only if it changes the outcome. Q2.1 and Q2.2
are deliberately separated, because a system can pass the first and fail the second,
and that combination localises the fault precisely.

### Q3. Bottleneck: when it fails, whose fault is it?

**Q3.1 How accurate is System 2 in practice?**
Measured against ground truth, how often is the emitted sub-goal the correct one?
And does per task accuracy predict per task success?

**Q3.2 Would a perfect System 2 help?**
This is the question that matters for deciding what to build next. If replacing
System 2 with an oracle does not improve success, then no amount of work on the
reasoning module can help, and effort belongs on System 1 instead.

### What the questions demand of the design

Answering these requires two architectural properties, both of which drove the
design in Section 3:

1. **The System 2 to System 1 interface must be swappable at evaluation time without
   retraining.** Otherwise every counterfactual costs a training run, and the
   oracle condition costs one that cannot be trained at all.
2. **The system must record what System 2 emitted at every control step**, so that
   detection can be measured separately from outcome.

---

## 3. System design

### 3.1 The interface is one discrete integer

System 2 emits a per task **phase** index, a small integer. The policy composes it
with the task identity into a single global row, remaps that row to a **skill**
class, and System 1 embeds the skill and conditions on it. For the bowl and drawer
task, the phases are 0 for "grasp the black bowl", 1 for "place the black bowl in
the bottom drawer of the cabinet", 2 for "close the bottom drawer of the cabinet".

This is the single most consequential design decision in the project, so the
alternatives deserve a proper accounting.

**The candidates.** Three interfaces are plausible for passing information from a
semantic module to a reactive one:

1. A **continuous latent vector**, produced by System 2 and consumed by System 1.
   This is what Figure's Helix and NVIDIA's GR00T N1 use, and it is the highest
   bandwidth option.
2. A **language embedding**, for example a sentence encoder applied to the current
   sub-goal string.
3. A **discrete index** into a fixed, finite set of sub-goals.

**Why the discrete index wins here.** Three properties were required, and only the
integer has all three.

*It is cheap enough to recompute every control step.* This is what makes the loop
closed rather than a one shot plan. A latent from a multi billion parameter VLM
cannot be recomputed at 10 Hz on one GPU alongside training runs; the selector used
here is a small head over features System 1 has already computed (161,988
parameters, Section 4.2), so it is effectively free. If System 2 can only run once
per episode, the entire error recovery question (Q2) is unanswerable by
construction, because the system cannot revise anything.

*It is swappable at evaluation time.* This is what turns three trained checkpoints
into seven conditions, and it is worth being concrete about why the alternatives are
not. Consider `A_shuffled`, which feeds a deliberately wrong sub-goal to test
whether System 1 attends to the channel at all. With an integer this is a one line
change: emit a different valid index. With a continuous latent there is no
principled construction of "wrong but plausible": a random vector is off manifold
and System 1's response tells you nothing about whether the channel carries meaning,
while a latent from a different timestep is not clearly wrong. The same applies to
`A_oracle`. Ground truth progress is naturally expressed as "the robot is in stage 2
of 3". Expressing ground truth as a latent vector would require inverting System 2,
which is not possible. **The discrete interface is what makes the ablation feasible
at all**, and the ablation is the deliverable.

*It is legible.* The emitted value can be drawn on a rollout video and read by a
human. This is not cosmetic. The bottleneck analysis (Q3) needs a per step
comparison of what System 2 believed against what was actually true, and with an
integer that comparison is exact equality. With a latent, "was System 2 right" has
no well defined answer without training a probe, and a probe introduces its own
error.

**What is given up.** Bandwidth. An integer from a 26 element vocabulary carries at
most about 4.7 bits per step. A latent carries hundreds of dimensions and could
express "the mug is on the left, slightly rotated, and the drawer is already open"
rather than merely "stage 2". Section 10.2 returns to this: the low bandwidth
interface is the main thing that would change in a Helix or GR00T style redesign.

**Why the index is not a bare per task phase.** With one shared embedding table,
phase 2 means "grasp the second object" in task 0 and "place the moka pot" in task 2.
Those two meanings would collide on one learned vector. ACT takes no language input,
so nothing downstream can separate them: System 1 cannot know which task it is in
except through the pixels, and the pixels of two different kitchens are not a
reliable task identifier. The policy therefore never conditions on a raw phase. It
composes (task, phase) into a global row first, and only then looks the row up.

Section 3.5 describes what those rows mean, which took two attempts to get right.

### 3.2 System 1: why ACT

System 1 is an Action Chunking Transformer (ACT), used essentially unmodified.

**Why a chunked action model rather than a single step policy.** LIBERO
demonstrations are smooth, multi second motions. A policy that predicts one action
at a time must rediscover temporal consistency at every step and tends to produce
jittery, compounding error trajectories. ACT predicts a chunk of 100 future actions
from one observation, which bakes temporal consistency into the output and is the
standard remedy for compounding error in behaviour cloning.

**Why ACT rather than a diffusion policy.** A diffusion or flow matching head models
multimodal action distributions better, which matters when several distinct motions
are all valid. Three reasons ACT was preferred here. It is the reference
implementation in LeRobot and is known to train stably at this data scale, which
removes a large source of "is the baseline broken or is the idea wrong" ambiguity.
It trains in about four hours per condition on one GPU, and the experimental design
needs several training runs plus a large evaluation budget. And its inference is a
single forward pass, which keeps the closed loop cheap: a diffusion head requires
multiple denoising steps per decision, which would have made recomputing System 2
every step a smaller fraction of a much larger cost.

**Why not a VLA with a language backbone.** A model that consumes the instruction
text directly would arguably subsume System 2. That is exactly why it was avoided:
if System 1 can read the instruction, the contribution of an explicit sub-goal
channel becomes impossible to isolate, and Q1 stops being answerable. Using a
language blind System 1 makes the sub-goal index the *only* route by which task
structure reaches the controller, which is what gives the ablation its clean
interpretation.

The cost of that choice is recorded honestly in Section 8.1: a language blind
System 1 cannot generalise to a new instruction, and the task embedding hard codes
ten tasks.

### 3.3 Offline decomposition is rule based, not an LLM

Sub-goals come from a rule based splitter grounded in the BDDL object names
(`scripts/01_make_subgoals.py`). The `libero_10` instructions are literally
compositional ("put A in B **and** close it"), so a regular expression split over
ten hand checkable tasks beats adding a model dependency plus a new failure mode.
The output is 35 sub-goals over 10 tasks, two minimum and four maximum per task,
and it was read once by hand before use.

An LLM would be the obvious choice at larger scale, where instructions are not
templated and hand checking is infeasible. At ten tasks it would introduce
nondeterminism and an API dependency into the one part of the pipeline that is
trivially verifiable.

The full decomposition, with the skill class of each sub-goal under each vocabulary,
is in Appendix A.5.

### 3.4 System 2: the selector

System 2 is a small classifier over features System 1 has already computed:

```
head( concat[ pooled_image_features(512), state_mlp(state)(64), task_embedding(32) ] )
    -> logits over max_phases (4)
```

Concretely: a two layer MLP over the 8 dimensional proprioceptive state (64 hidden,
64 out), a 32 dimensional embedding of the task identity, both concatenated with the
512 dimensional globally average pooled feature map from ACT's own ResNet18 backbone,
and a two layer head (256 hidden, dropout 0.1) emitting four logits. Total: 161,988
parameters.

Four design points, each of which affects a measurement:

**It reuses ACT's vision features rather than running a second backbone.** That is
what makes recomputation at every control step affordable. It takes the third person
view only: the wrist camera view is mostly gripper and table surface, which is weak
evidence for "how far through the task am I".

**Task identity is given, not inferred. Only progress is inferred.** This is a
deliberate simplification: inferring the task as well would confound Q3, because a
wrong sub-goal could mean either "wrong stage" or "wrong task", and the two have
different remedies. At evaluation, the task id is recovered from the instruction
string that LeRobot places in the batch, which is the only task identity the standard
policy interface exposes.

**It has no memory and no monotonicity constraint, by design**, so that a backward
transition remains available as the error recovery signal that Q2.1 measures. Section
6.4 reports the measured consequence of the no memory choice.

**The logits are masked, not clamped**, to the valid phases for the current task.
Tasks have two, three or four phases while the head always emits four. Clamping an
out of range argmax down to the last valid phase would invent a confident answer the
model never gave, and would corrupt the agreement statistic used for Q3.1. Masking
asks the real question: which valid phase scores highest?

### 3.5 What the integer *means*: row numbers against a skill vocabulary

This took two attempts, and the first one was wrong in a way worth recording.

#### Attempt 1: number the (task, phase) pairs

`SubgoalIndex` flattens `(task_id, phase)` into 35 contiguous global rows plus a null
row, 36 in total, by the affine rule `to_global(t, p) = offset[t] + p`. That removes
the collision described in Section 3.1. It also introduces two defects.

*The channel is not Markovian.* Row 26 does not mean "grasp the cream cheese box".
It means "grasp the cream cheese box, **and** you are in task 7, **and** you are at
phase 2". It carries task identity and episode position on top of the instruction.
That is exactly the "which sub-goals came before and which come after" information a
dual system interface is supposed not to carry, since the point of the interface is
to tell System 1 what to do now, not where in a script it sits. A conditioned policy
can in principle read progress off the row number without ever reading the
instruction the row encodes.

*The training signal is fragmented.* Semantically identical sub-goals get separate
embedding vectors trained on disjoint subsets of the data. Six of the 35 rows are
literal duplicates of another row's description, so "grasp the white mug" is learned
twice, once as row 14 in task 4 and once as row 20 in task 6, from roughly half the
data each. On a 379 episode training subset that halving is not free.

#### Attempt 2: number the skills

The fix is a **canonical vocabulary**: a table mapping each of the 35 global rows to
a skill class, so that rows meaning the same thing share one embedding row.

| vocabulary | classes | embedding | rule |
|---|---|---|---|
| `v25` | 25 | (26, 64) | merges descriptions equal once articles and "and" are normalised away, then additionally strips ordinals |

`v25` performs its merges deterministically from exact text matches after
normalisation, so no judgement call is involved. The result: "grasp the moka pot" in
task 2 phase 1, "grasp the first moka pot" in task 8
phase 0, and "grasp the second moka pot" in task 8 phase 2 all become one class.
Task 8's classes under `v25` are `[9, 10, 9, 10]`, which means System 1 receives the
**same** instruction for both moka pots and must use the image to decide which one is
still on the table. That is the sharpest available test of whether the channel
carries a skill or a position, and it is only constructible because the vocabulary is
explicit.

The vocabulary derives deterministically from `build_vocabularies()` in
`scripts/01_make_subgoals.py`. No model is involved, which keeps the property Section
3.3 was chosen for: the decomposition remains trivially verifiable by hand.

**The shipped system uses `v25`.** The 25 skill vocabulary plus a null class gives a
26 row embedding table.

#### Options that were rejected

1. **Leave the code unmerged.** Zero work, keeps both defects.
2. **Tie the embeddings while keeping the 35 row numbering.** **Chosen.** See below.
3. **Renumber to a real 25 row vocabulary end to end.** Conceptually cleaner, since
   the logged index would itself be semantic. Rejected because it breaks four things
   at once. `to_global` stops being affine, so the vectorised
   `offsets()[tid] + phase` composition used by the training loops has to go.
   `to_local` becomes one to many. And, critically, the forward and backward
   transition counters compare `row > prev_row` to mean "forward progress", which a
   non monotone vocabulary makes meaningless. Those counters produce the error
   recovery headline number in Section 6.5, and the risk of silently corrupting them
   was not worth prettier logs.
4. **Use a language model or sentence embeddings to decide equivalence.** Rejected as
   unnecessary and actively risky. Exact string keys after normalisation already find
   every real merge, while a cosine similarity threshold would wrongly merge "place
   the white mug on the left plate" in task 4 with "place the white mug on the plate"
   in task 6, which differ in their target and must stay apart. A threshold that
   avoids that merge would also avoid the merges that are wanted.
5. **Widen the interface to two integers**, the skill class plus the per task phase.
   This is the natural repair for the one real cost of merging, that position
   information is destroyed, and it is cheap: the phase is already what System 2
   predicts, so System 2's job would not change, and a second gather table indexed by
   the    same global row would carry it. Measured, the factored pair `(v25, phase)`
   separates 32 of the 35 situations using 31 embedding rows.

   **Rejected, because the second integer carries no information System 1 does not
   already have.** ACT receives images and proprioceptive state and no text, so it
   reads the scene from pixels. Every one of the eight classes that `v25` collapses is
   visually distinguishable at the moment the distinction matters. In task 8 the first
   and second moka pot differ by whether one pot is already on the stove. Task 7 phase
   2 differs from task 1 phase 0 by whether the alphabet soup is already in the basket.
   The remaining collapses are across different tasks, where the objects and the layout
   differ outright. Supplying the phase would also defeat the experiment: the purpose of
   `v25` is to force vision to resolve which moka pot is meant, and returning the phase
   removes exactly that requirement.

   The phase does matter, in System 2 rather than System 1. System 2 predicts it, is
   scored on it, and every diagnostic in Section 6 lives in phase space: oracle
   agreement, forward and backward transition counts, and the sub-goal at failure. It
   is already where it belongs.

#### Where the remap lives, and why that placement matters

The remap is applied inside `SubgoalLatentProjection` in `modeling_subgoal_act.py`,
**not** where rows are composed. The module takes a `canon_table` of length 36 instead
of an `n_rows` integer, registers it as a non persistent buffer, sizes
`nn.Embedding(canon.max() + 1, embed_dim)`, and gathers `classes = canon[rows]` before
embedding.

The consequence is that everything outside that one module keeps speaking the 35 row
language. `index.name(row)`, both transition counters, and every logged index trace
stay correct with no edit at all, which is precisely the breakage option 3 would have
caused. The null row maps through the same table, so `config.null_index` stays at 35
and cannot index past a shortened table. Applying the remap at row composition time
instead would have made `null_index` an out of bounds gather: an `IndexError` on CPU,
and on CUDA a device side assert raised far from its cause.

#### What merging costs

Merging **removes task identity from the conditioning channel**. Under `v25`, tasks 0
and 7 share their phase 0 class, as do tasks 4 and 6. Three consequences follow and
all of them are reported:

* `B_static`, which encodes pure task identity, is a **weaker baseline by
  construction** under a merged vocabulary.
* Part of any conditioned advantage over `C_none` may have been task identity rather
  than progress tracking, which merging separates: under `v25` the channel no longer
  identifies the task.
* **Merged scores may come out lower.** This is a cleaner experiment, not a better
  score; merging changes the interface's semantics rather than being an improvement
  claim.

Section 6.2 reports what actually happened.

### 3.6 How the integer reaches ACT without forking it

ACT assembles its encoder token list inside `ACT.forward`, so adding a genuinely new
token would mean copying that method into this repository, which the LeRobot
documentation explicitly warns against and which would make every upstream fix a
manual merge.

Instead the plugin wraps the smallest upstream unit the change actually touches:
`encoder_latent_input_proj`, the Linear layer that produces the first encoder token
from the VAE latent. The projected sub-goal embedding is **added** to that token.

This is not a compromise at inference time. ACT feeds a zero latent when not
training, so at rollout that token carries the sub-goal and nothing else. During
training it carries the sum of the VAE latent and the sub-goal, in the same way
positional embeddings are added rather than concatenated.

**The cost, stated plainly.** Addition is a weaker conditioning mechanism than FiLM,
which would let the sub-goal modulate the vision features multiplicatively rather
than merely offset one token. A check was defined before any result was collected: if
`A_shuffled` ever ties `A_learned`, System 1 has learned to ignore this channel and
FiLM is the first thing to try. That check passed decisively (Section 6.1), so
addition was retained.

### 3.7 Action chunking: predict 100, execute 10, flush on change

ACT defaults both `chunk_size` and `n_action_steps` to 100. That default was tuned
for ALOHA at 50 Hz, where 100 steps is two seconds of motion. LIBERO runs at 10 fps,
so the same number is ten seconds, and a 520 step episode then contains only about
five policy decisions. System 2 can recompute every step, but its answer only reaches
System 1 when a new chunk is requested, so at 100 the closed loop this project exists
to demonstrate cannot appear at all, and Q1.2 and Q2 become unanswerable.

`chunk_size` stays at 100, because predicting far ahead is what action chunking is
for and executing only a prefix is ACT's normal deployment pattern. `n_action_steps`
is set to 10, giving about 52 decisions per episode. It is an inference only
parameter, so it is identical across every condition and can be applied to
checkpoints trained before it was chosen.

The action queue is additionally **flushed whenever the sub-goal changes**, which
bounds System 2 latency at one step instead of ten. One subtlety: the queue holds one
batched tensor, so a flush is all or nothing and cannot be done per episode. Keying it
off element 0 meant that under batched evaluation (the sweep runs
`eval.batch_size=10`) episode 0 decided when the other nine replanned. The flush
therefore triggers whenever **any** element's sub-goal moves.

This choice was validated by accident. When the perturbation harness was found to be
running at the baked in 100, its unperturbed control scored 30 percent against 45
percent at `n_action_steps=10`, on the same checkpoint and the same episodes.

### 3.8 Hysteresis over System 2's output

Added after the ablation revealed instability rather than inaccuracy (Section 6.4). A
competing phase is committed only after it wins N consecutive steps, asymmetric at 3
forward and 8 backward.

The asymmetry is principled. Moving forward is the normal flow of the task and
delaying it costs only a few steps. Moving backward discards progress, so it should
require stronger evidence. But it is **not blocked**: measured against the oracle,
roughly half the backward motion on failed episodes is legitimate, because a dropped
object genuinely does move the task backward. A hard monotonic ratchet would have
deleted the Q2.1 signal rather than cleaning it up, which would have made the system
look better while destroying the measurement.

It is inference only, applies to the learned selector only so the other conditions
remain untouched as controls, and defaults to (1, 1), which commits immediately and
therefore reproduces the memoryless selector exactly.

---

## 4. Implementation

### 4.1 Out of tree plugin, never a fork

`policy/` (installed as `lerobot_policy_subgoal_act`) is a separate installable
distribution, a sibling of `scripts/` rather than a directory inside a modified
LeRobot checkout. This keeps the work a plugin that a LeRobot upgrade cannot silently
break, instead of a divergent copy.

LeRobot discovers such plugins in `lerobot/utils/import_utils.py`: it scans
`importlib.metadata.distributions()` for the `lerobot_policy_` prefix and then calls
`importlib.import_module(dist_name)`, using the distribution name **directly as a
module name**. The `pyproject.toml` project name therefore uses underscores, against
the usual PyPI hyphen convention. The import is wrapped in a bare `except Exception`,
so a hyphenated name disables the plugin silently, and the only symptom is
`--policy.type=subgoal_act` being reported as an unknown policy type with a list of
built in types that points nowhere near the cause.

A second discovery trap: LeRobot registers third party plugins inside the
`lerobot-train` and `lerobot-eval` entry points, not on `import lerobot`. Any
standalone script that loads a `subgoal_act` checkpoint has to call
`register_third_party_plugins()` itself, or `PreTrainedConfig.from_pretrained`
rejects the checkpoint with "policy type not registered" and a list of built ins.

### 4.2 Parameter budget

| component | parameters | what it is |
|---|---|---|
| System 1, ACT | 51,600,263 | ResNet18 vision backbone, transformer encoder and decoder, action head emitting 100 x 7 actions |
| Skill embedding table | 1,664 (26 x 64) | one learned vector per skill class, 25 real plus one null |
| Skill projection | 33,280 | Linear mapping the embedding into ACT's token width |
| System 2 selector | 161,988 | state MLP, task embedding, two layer head over pooled vision features |
| **conditioning pathway, total** | **196,932** | the three rows above, under `v25` |
| **total learnable** | **51,797,195** | measured, logged at training start |

The conditioning pathway costs **0.38 percent** of the model.

The vocabulary changes only the embedding row of that table. Measured total learnable
parameters, taken from the training logs:

| vocabulary | conditioning pathway | total learnable |
|---|---|---|
| `v25` | 196,932 | 51,797,195 |

The single `v25` vocabulary therefore determines the pathway's parameter count, and
any difference measured is a difference in what the
channel means, not in how much capacity the model has.

Two things follow from this table and both matter for interpreting the results.

First, the interface is extremely cheap. Whatever System 2 contributes, it
contributes for under half a percent of the parameters, so a small measured gain
would still have been a good trade. The finding in Section 6 is not that the module
is too expensive; it is that it does not change the outcome.

Second, **all conditions report the same parameter count for a given vocabulary**,
because the plugin instantiates the embedding table and the selector regardless of
the training mode. `C_none` is therefore `subgoal_act` with the conditioning pathway
present but unused, not literally stock ACT. This is the honest description of the
floor condition, and it slightly favours a conservative reading: `C_none` carries a
small number of parameters it never uses, so it is if anything marginally handicapped
relative to true stock ACT.

### 4.3 Training labels

LIBERO ships no sub-goal labels, so they are derived from the demonstrations
themselves. The signal is the gripper: in pick and place, sub-goal boundaries line up
almost exactly with the gripper opening and closing. Measured on this dataset, the
gripper aperture (`observation.state[6] - state[7]`) is cleanly bimodal at roughly
0.006 closed and 0.075 open.

A fixed absolute threshold does **not** work, and getting that wrong is silent, so the
thresholds are derived **per episode** relative to that episode's own open to close
span: the state flips to closed below 35 percent of the span and back to open only
above 65 percent, with the open reference floored at the known stable fully open value
of 0.080. That hysteresis band is what turns a noisy continuous signal into
transitions with no chatter.

Phase advances as

```
phase = 2 * (placements completed) + (1 if currently gripping)
```

Output is a sidecar parquet joined on the global frame index, deliberately not a new
column in the dataset. Rewriting the LeRobot dataset would mean regenerating
`meta/stats.json`, and an error there corrupts normalization for every condition
silently. A sidecar is reversible and cannot poison the cached dataset.

**379 episodes and 101,469 frames were labelled.** 24 percent of episodes have a
segment count that disagrees with their sub-goal count, concentrated on tasks 0, 1, 6,
7 and 8. Some disagreement is expected and benign: a regrasp adds transitions, and
"turn on the stove" involves no grasp at all. But the concentration on specific tasks
is a real weakness, carried into Section 8.2 as a limitation, because those are
largely the tasks the selector performs worst on and cause and effect are not
separated here.

**Training uses teacher forcing.** System 1 is trained on the ground truth sub-goal,
not System 2's prediction. This stops an early, near random selector from corrupting
System 1, and it keeps the two losses independent so a later failure can be attributed
to one of them. The cost is exposure bias at evaluation: part of any
oracle against learned gap is exposure bias rather than System 2's quality.

**Sub-goal dropout at 10 percent** replaces the ground truth row with the null row
during training. Without this the null row would never receive a gradient, and
`C_none`, which relies on it at evaluation, would be reading a randomly initialised
vector: measuring noise rather than an architecture.

The selector's cross entropy is added to ACT's loss at weight **0.5**. The two are in
incommensurable units (metres and radians against nats) and share no trunk, so this
only sets their relative effective learning rates.

### 4.4 Relabelling with a VLM: a negative result worth recording

**Why it was attempted.** The gripper heuristic is provably wrong on at least one
task. On task 2 it advances the label on the gripper transition that turns the stove
knob, while the truth is that phase 0 holds until `turnon(flat_stove_1)` becomes true.
A frozen local Vision Language Model, Qwen2.5-VL-7B, was therefore used to relabel
training frames directly from the image, on the reasoning that a model that can see
the scene should not make that particular mistake.

**It did not work.** Agreement with the gripper labels was 24.3 percent, which is
chance for a four phase task, because the model returns a near constant phase for an
entire episode.

**Three explanations were ruled out empirically**, which is what makes this a result
rather than a bug report:

* Not plumbing. The identical call path names solid red and solid blue images
  correctly, so the image reaches the model and the answer comes back parsed.
* Not static frames. The mean absolute pixel delta between sampled frames is about 15
  out of 255, so the frames genuinely differ.
* Not resolution or viewpoint. Adding the wrist camera and upscaling from 256 to 448
  moved agreement by 0.1 points and left the per task numbers identical.

What remains is the model's grounding on 256 x 256 simulator renders. A frozen model
decomposes compositional instructions near perfectly and grounds badly on small
simulator images. The two halves of the same model are not equally reliable, and the
pipeline uses only the half that is.

**What was done about it.** The VLM labelled condition was dropped rather than trained
on chance level labels. Training it would have consumed a four hour slot to measure
label noise while presenting as an architecture result, which would have made the
ablation misleading.

**The related argument, against putting a VLM in the control loop at all.** At ten
conditions, ten tasks, ten episodes and 520 steps, a single evaluation sweep is
roughly 364,000 System 2 calls. A frozen large model at that call volume is not
affordable here, and, more importantly for this study, it cannot be shuffled, frozen
or jointly trained the way the small selector can. The four conditions that carry the
ablation all depend on manipulating System 2 cheaply and precisely. Swapping in a
large frozen model would confound model scale with the thing being measured.

---

## 5. Evaluation design

### 5.1 Dataset and benchmark

| item | value |
|---|---|
| Dataset | `lerobot/libero`, revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| Full dataset | 1,693 episodes, 273,465 frames, 40 tasks, 10 fps |
| Training subset | 379 episodes, `libero_10` only |
| Benchmark | LIBERO-Long (`libero_10`), 10 tasks, 520 step cap |
| Simulator | `huggingface/lerobot-libero`, robosuite 1.4.0, MuJoCo 3.8.1, EGL rendering |

The dataset revision is pinned and every training and evaluation command passes it
explicitly. This is not bureaucratic. Hub datasets can be re uploaded in place, and a
success rate is only comparable against a single revision; a silent dataset change
between two conditions would produce a difference that looks architectural and is not.

**Why the training subset is restricted to `libero_10`.** The full dataset holds 1,693
episodes across 40 tasks, and using all of them while evaluating on ten would handicap
every condition. The first overnight baseline did exactly this and scored 1 out of 100.
A floor condition has to be a fair floor, otherwise Q1.1 is answered by a training
artefact rather than by architecture.

One qualification. Under `v25`, merged classes are shared across tasks, so the
conditioned conditions receive only the instruction through the sub-goal index, not the
task. The argument for restricting the subset is
unaffected, since it rests on all conditions facing the same ten tasks.

### 5.2 Baselines

Two baselines are used, and they answer different questions. Confusing them is easy
and would weaken the report, so they are described separately.

**`C_none`, the floor.** No conditioning at all: System 1 receives images and
proprioception and nothing else. The sub-goal pathway exists in the network but is
never fed, so the policy must infer task and progress entirely from pixels.

This is the condition the architecture must beat in order to justify existing. If a
dual system policy cannot outperform the same controller with the second system
deleted, then the second system contributes nothing and the added complexity is
unjustified. It is the reference point for **Q1.1**.

**`B_static`, the brief's named naive alternative.** One fixed instruction embedding
for the entire episode: the per task first sub-goal, held constant. It represents the
obvious cheap approach, namely "tell the policy what the task is, once, and let it get
on with it". It has task information but no progress information.

This is the reference point for **Q1.3**. The gap between `C_none` and `B_static`
measures the value of task identity, and the gap between `B_static` and `A_learned`
measures the value of progress tracking. Under `v25` the first of those readings
weakens, because merged classes no longer identify
the task.

**Why both baselines get their own training run.** This costs two extra training runs
of about three and a half hours each, and the alternative was tempting: train one
conditioned policy and simply feed it a constant embedding, or nothing, at evaluation
time. That would have been wrong. A policy trained with a live, changing sub-goal
signal and then given a frozen or absent one at evaluation is operating out of
distribution, and would fail for reasons that have nothing to do with the architecture
being tested. It would flatter `A_learned` dishonestly. A fair counterfactual trains
each baseline properly, so that each is the best version of itself.

There is a second, subtler reason. An embedding row that is never fed during training
keeps its random initialisation. A condition that relies on such a row at evaluation is
measuring noise, not an architecture. This is also why the configuration refuses to
evaluate a checkpoint trained with no conditioning under any other evaluation mode.

### 5.3 Evaluation protocol and statistics

Evaluation is **paired** throughout. `--env.init_states=true` combined with an
identical seed means every condition faces the same scene layouts, in the same order,
with the same object placements. The question therefore becomes "did condition A
succeed on the episodes where condition B failed", rather than a comparison of two
independently sampled averages.

This matters more than it might appear. LIBERO initial states vary considerably in
difficulty, and an unpaired comparison of two conditions at 100 episodes each is
dominated by which condition happened to draw easier layouts. Pairing removes that
variance source entirely, roughly halving the effect size needed to claim a real
difference, and it costs nothing beyond setting a flag.

**What one episode is.** The simulator is reset to a stored initial state, which fixes
where every object starts. The policy runs until it either satisfies the task's BDDL
goal predicates (a success) or reaches the 520 step cap (a failure). There is no
partial credit: the outcome is a single binary value. This is the standard LIBERO
protocol and it is why all statistics here are statistics of proportions.

**What a seed controls.** A seed fixes the generator that selects which stored initial
states are used and in what order, together with any stochastic element of the
environment. It does **not** control model initialisation: all checkpoints were trained
once. The seeds vary the evaluation sample, not the trained weights, and Section 8.2
records this as the single largest limitation.

**How many seeds, and why three.** Each condition was evaluated at three seeds (1000,
2000, 3000), 100 episodes per seed, giving 300 episodes per condition.

* One seed is not enough. A single 100 episode run has a 95 percent interval of roughly
  plus or minus 10 points, which is wider than most of the differences under test. The
  observed per seed spread bears this out: `A_debounced` scored 54, 46 and 51 percent
  on the three seeds, an 8 point range from identical weights.
* Three seeds at 100 episodes each narrows the interval to roughly plus or minus 5.6
  points, which is enough to separate the large effect (`A_shuffled`) from the rest,
  though not enough to separate the conditions clustered in the middle.
* More seeds would have been better and were not affordable.

**Wilson intervals**, not the normal approximation, because the normal interval can
extend outside [0, 1] and understates asymmetry at these sample sizes.

**McNemar's exact test** on discordant pairs, which is the correct test for paired
binary outcomes: it looks only at episodes where the two conditions disagreed, and asks
whether the disagreements are lopsided.

Note that Wilson intervals are computed per condition and are therefore *unpaired*,
which makes them conservative for this design. The paired test has more power; both are
reported, and where they disagree the paired test is the more appropriate one. **The
important consequence: overlapping confidence intervals do not prove two conditions are
the same.**

**Two things deliberately not done.** No hyperparameter search was run per condition,
since that would optimise conditions unequally. And no condition was evaluated more
times than another and then reported at its best, which is the most common way an
ablation becomes untrue.

### 5.4 The oracle, and why it is an honest upper bound

Per predicate BDDL goal state is readable on all ten tasks
(`env.env.parsed_problem["goal_state"]` combined with `env.env._eval_predicate`), so
the oracle condition uses ground truth rather than a heuristic proxy. This is what
makes Q3.2 answerable at all.

Two caveats must be carried:

* It is a **goal** oracle, not a fine grained sub-goal oracle. There are roughly two
  predicates per task, and nothing at all for intermediate states such as "the bowl is
  currently in the gripper". The oracle knows when a stage has been *completed*, not
  everything a perfect reasoner would know.
* Some predicates are **already true at t=0**, for example task 8's
  `turnon(flat_stove_1)`. The t=0 vector is therefore treated as a baseline rather than
  as zero, otherwise both labelling and progress metrics are wrong from the first step.

`SubgoalOracle` anchors on ground truth wherever BDDL provides it, and uses the gripper
only for the grasp versus place distinction that BDDL cannot express. That half is
**proprioception**, which the robot legitimately knows about itself, rather than
privileged simulator state. The condition therefore remains an honest upper bound on
*progress knowledge*, rather than smuggling in knowledge of the answer. This
distinction is what lets `A_oracle` be interpreted as "the best any System 2 could do"
instead of "cheating".

**A rejected oracle design, recorded because the failure would have been silent.** An
earlier version fed System 1 a sub-goal derived from a count of satisfied BDDL goal
predicates. The training labels advance on gripper transitions, so for a pick and place
pair they read `2 * (placements completed) + (1 if currently gripping)`. A predicate
count sees only the first term. It therefore lags a full phase, and on tasks 0, 1, 4, 6,
7 and 8 it can never reach the top sub-goal index at all. Conditioning System 1 on it
would mean the oracle condition speaks a language System 1 was never trained in, which
understates the reasoning ceiling and can invert the answer to Q3.2, the single most
important question in the study.

A related trap: `external_phase` is a **per task phase**, not one of the global rows.
Building the oracle with the global count would let it emit indices that map straight to
the null row, silently unconditioning the policy for most of the episode.

### 5.5 Instrumentation

Full time series are logged per episode rather than summary statistics. Re running a
sweep is not affordable, and the analysis performed after a sweep routinely needs
quantities that were not anticipated before it. Every episode records:

* the per step sub-goal trajectory (`subgoal_index_trajectory`),
* forward and backward transition counts,
* the sub-goal held at the moment of failure (`subgoal_at_failure`),
* per step agreement with the oracle (`s2_oracle_agreement`),
* steps to success, and the per predicate first satisfaction step.

This is what makes Q2.1 separable from Q2.2. Success alone cannot distinguish "the
system never noticed the disturbance" from "the system noticed and could not act on
it", and those two diagnoses point at completely different next steps.

Every rollout video carries **two** sub-goal readouts, System 2's choice and the
oracle's. This is a measurement decision rather than a presentation one. Agreement plus
failure is an execution failure; divergence before failure is a reasoning failure; one
line alone cannot separate them. The oracle line costs nothing, because the predicates
are already being evaluated for the success check.

### 5.6 The conditions

Seven conditions from three trained checkpoints. Only three require training; the rest
swap the System 2 to System 1 interface at evaluation time, which is precisely what the
discrete interface (Section 3.1) was chosen to enable.

| condition | System 2 source | trained | answers | compared against | a difference means |
|---|---|---|---|---|---|
| `A_learned` | selector, recomputed every step | yes | the full system, reference for all comparisons | everything else | this is the system under test |
| `C_none` | nothing | yes | Q1.1 | `A_learned` | the sub-goal channel contributes (or does not) |
| `B_static` | one fixed instruction embedding | yes | Q1.3 | `C_none` and `A_learned` | task identity has value; progress tracking has value beyond identity |
| `A_shuffled` | deliberately wrong sub-goal | no, reuses A | Q1.1 | `A_learned` | System 1 genuinely attends to the channel; also a positive control on the protocol |
| `A_frozen` | fixed time schedule, open loop | no, reuses A | Q1.2, Q2.1 | `A_learned` | closing the loop has value beyond having a decomposition |
| `A_oracle` | ground truth from BDDL predicates plus gripper | no, reuses A | Q3.2 | `A_learned` | the headroom available to any better System 2 |
| `A_debounced` | selector plus hysteresis | no, reuses A | Q3.1 follow up | `A_learned` | selector instability, as distinct from inaccuracy, matters |

The frozen schedule is `phase = min(step * n_phases // max_episode_steps, n_phases - 1)`,
which spreads the phases evenly over the 520 step cap.

Three of these deserve elaboration, because their interpretation is not obvious.

**`A_shuffled` serves two roles.** Its primary role is to answer Q1.1 from the
opposite direction: rather than removing the signal, it corrupts it. But its second
role is methodological and arguably more important. It is a **positive control**. If it
had scored the same as `A_learned`, then every other null result in this study would be
uninterpretable, because the protocol would have demonstrated no ability to detect a
difference of any size. Its large effect is what licenses reading the other comparisons
as genuine nulls rather than as an insensitive experiment.

**`A_frozen` is the control that carries the recovery argument.** It follows a clock and
is **structurally incapable** of moving its sub-goal backwards. A difference between it
and `A_learned` under perturbation is therefore demonstrated by construction rather than
by a p value: there is no possibility that `A_frozen` recovered by some unmodelled
route, because the mechanism does not exist in it. This is a stronger form of evidence
than a statistical comparison between two systems that both could in principle recover.

**`A_oracle` sets the ceiling, and its null result is the most decisive in the study.**
If an oracle does not beat the learned selector, no achievable System 2 will either, and
the question is closed.

### 5.7 Which condition answers which question

| question | conditions compared | what the comparison shows |
|---|---|---|
| Q1.1 conditioning matters | `A_learned` against `C_none`, and `A_shuffled` against `A_learned` | whether the channel carries usable signal at all |
| Q1.2 online selection matters | `A_learned` against `A_frozen` | whether closing the loop adds anything beyond the decomposition |
| Q1.3 naive baseline suffices | `B_static` against `C_none` and against `A_learned` | separates task identity from progress tracking |
| Q2.1 disturbance detected | backward transitions, `A_learned` against `A_frozen` | whether the mechanism activates, independent of outcome |
| Q2.2 detection aids recovery | success under perturbation, all three | whether activation changes the result |
| Q3.1 System 2 accuracy | agreement with oracle, per task | how good the reasoning actually is, and whether it predicts success |
| Q3.2 perfect reasoning helps | `A_oracle` against `A_learned` | whether any better System 2 could help |

### 5.8 Two measurement paths, and why numbers differ between them

`A_oracle` cannot run through `lerobot-eval` at all: it requires privileged simulator
state (BDDL predicate truth) that the standard policy interface never exposes. The
policy raises rather than guessing. It is therefore driven by a purpose built rollout
harness (`scripts/08_video.py`) which sets the phase externally at every step.

That harness constructs its own simulator instance, and its numbers are **not
interchangeable** with `lerobot-eval` numbers. The same `A_learned` checkpoint scores
50.7 percent through `lerobot-eval` over 300 episodes and 60.0 percent through the
harness over 30. Both tables are therefore reported, every quoted figure states its
source, and comparisons are only ever made **within** a table, never across them.

The perturbation study also runs in its own harness loop, for the same reason: it has
to reach into the simulator to inject a disturbance.

### 5.9 Compute environment and cost

All training and all evaluation reported here ran on a single machine.

| component | specification |
|---|---|
| CPU | Intel Core i9-14900KF, 24 cores (8 performance, 16 efficiency), 32 threads |
| CPU clock | 6.0 GHz maximum stock, **administratively capped to 4.5 GHz** |
| Memory | 125 GB |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| NVIDIA driver | 550.144.03 |
| Operating system | Ubuntu 22.04.5 LTS, kernel 6.8.0-136 |
| Container runtime | Docker 29.7.2, Compose v5.4.0, NVIDIA Container Toolkit 1.20.0 |
| Container image | 31 GB, built on the machine rather than pushed to it |

**The clock cap is a documented experimental condition, not an operational footnote.**
The i9-14900KF is a Raptor Lake part with well documented instability triggered by high
voltage at high single core boost, rather than by aggregate load. Uncapped, this machine
produced segfaults on two different physical cores, a crash inside `libnvrtc.so.12`
(NVIDIA's runtime compiler) during a training run, four kernel oops dumps, and
unkillable processes in uninterruptible sleep requiring a hard reboot. Because the
trigger is voltage rather than utilisation, the usual mitigations do nothing: a cgroup
CPU quota, `nice`, and core pinning all leave the failure mode intact, since a single
threaded compilation boosting to 5.7 GHz is precisely the failure case and is a *low*
load. Only the frequency cap addresses it. Every number in this report was produced
under the cap, and exactly one kernel fault occurred across the entire campaign, during
a window after a reboot before the cap had been reapplied.

**Training cost**, measured, from the training logs:

| condition | vocabulary | steps | batch | wall clock | final loss | selector acc |
|---|---|---|---|---|---|---|
| **`A_learned` (shipped)** | **v25** | **100,000** | **64** | **4 h 06 m 21 s** | **0.129** | **0.999** |
| `B_static` (shipped) | v25 | 100,000 | 64 | 3 h 24 m 35 s | 0.118 | n/a |

Roughly 63 epochs over the 379 episode subset, at 6.8 to 8.5 optimisation steps per
second. The conditioned runs are slower because they additionally train the selector
and compute its loss. Evaluation throughput was 2.8 to 4.6 seconds per episode of wall
clock time at `eval.batch_size=10`, since ten episodes run in parallel.

**Experimental validity rule.** Conditions were never split across machines. Different
GPU, different VRAM, and the temptation to raise the batch size on a larger card all
confound the comparison: if one condition trains at batch 96 and another at batch 64,
the headline number measures hardware rather than architecture.

**Deployment.** Code travelled by git so every run sits at an identifiable commit;
results returned by rsync, since checkpoints are roughly 591 MB each; jobs ran as
detached Docker containers so a dropped SSH session does not kill a run and the exit
code and log survive the process.

---

## 6. Results

All results in Sections 6.1 to 6.7 come from one campaign: all conditions trained and
evaluated at a fixed budget of 100,000 steps, on one machine, at the same batch size.
Section 6.8 reports a separate campaign at 200,000 steps and is labelled as such.

### 6.1 Main ablation, `lerobot-eval`, 300 paired episodes per condition

| condition | seed 1000 | seed 2000 | seed 3000 | pooled | 95% Wilson CI | McNemar p vs `A_learned` |
|---|---|---|---|---|---|---|
| `A_learned` | 51.0 | 51.0 | 50.0 | **50.7** | [45.0, 56.3] | reference |
| `A_debounced` | 54.0 | 46.0 | 51.0 | 50.3 | [44.7, 56.0] | 1.00 |
| `A_frozen` | 48.0 | 53.0 | 49.0 | 50.0 | [44.4, 55.6] | 0.89 |
| `C_none` | 43.0 | 49.0 | 47.0 | 46.3 | [40.8, 52.0] | 0.27 |
| `B_static` | 43.0 | 45.0 | 44.0 | 44.0 | [38.5, 49.7] | 0.06 |
| `A_shuffled` | 37.0 | 32.0 | 33.0 | 34.0 | [28.9, 39.5] | 7.8e-08 |

Paired McNemar counts against `A_learned`, which is what the p values are computed
from:

| comparison | `A_learned` only | other only | p |
|---|---|---|---|
| vs `A_shuffled` | 69 | 19 | 7.8e-08 |
| vs `B_static` | 61 | 41 | 0.059 |
| vs `C_none` | 65 | 52 | 0.267 |
| vs `A_frozen` | 29 | 27 | 0.894 |
| vs `A_debounced` | 50 | 49 | 1.000 |

**Three things to read from this table.**

First, **`A_shuffled` is clearly separated from everything else**. Its interval,
[28.9, 39.5], does not overlap `A_learned`'s, and the paired test gives 69 episodes won
against 19 lost. A deliberately wrong sub-goal costs 16.7 points relative to the correct
one. This is the result that establishes the conditioning pathway is real, and it is
what makes the surrounding nulls interpretable: the protocol at this sample size **can**
detect an effect when one exists.

Second, **the top four conditions are not separated**. `A_learned` at 50.7, `A_debounced`
at 50.3, `A_frozen` at 50.0 and `C_none` at 46.3 have intervals that overlap almost
completely, and no paired test among them approaches significance. On the criterion
stated in Section 5.3, the honest statement is that these four conditions are
indistinguishable at this sample size. The 4.4 point nominal advantage of `A_learned`
over `C_none` rests on 117 informative episodes split 65 to 52, which is not far enough
from a coin flip to call.

Third, **`B_static` at 44.0 sits below `C_none` at 46.3**. The brief's named naive
baseline performs worse than no conditioning at all.

**The asymmetry is the finding.** A wrong sub-goal costs 16.7 points; a correct one
gains at most 4.4 and possibly nothing. ACT already infers the stage from the images, so a
correct token is largely redundant, while a contradictory one overrides the camera.

### 6.2 The positive control on the merged vocabulary

The shipped `v25` vocabulary is the only one used in the campaign, and every trained
condition runs on it. The positive control separates cleanly on the merged channel:
`A_shuffled` under `v25` scores 34.0 percent against `A_learned` at 50.7 percent,
paired p below 1e-7. System 1 therefore reads the merged vocabulary, and the
separation that Section 6.1 establishes for the conditioning channel is not an
artefact of one particular set of embedding rows.

### 6.3 The oracle ceiling, on the common harness

`A_oracle` cannot appear in the table above, and a ceiling that cannot be compared to
anything is useless. It is therefore measured through the rollout harness, alongside
`A_learned` on the **same 30 episodes** (3 episodes on each of the 10 tasks).

| condition | success | oracle agreement |
|---|---|---|
| `A_learned` | 18/30 = 60.0% | 62.8% |
| `A_oracle` | 16/30 = 53.3% | 100% by construction |

Paired: `A_learned` wins 4 episodes, `A_oracle` wins 2, p = 0.69.

**Replacing the learned selector with ground truth does not improve success.** It is
nominally worse here, though at 30 episodes the interval is roughly plus or minus 17
points, so the supported statement is "no benefit detected", not "the oracle is worse".
This is the direct answer to Q3.2 and it is negative.

The result is not fragile. It is reproduced independently at a different training budget
and a larger sample: on the 200,000 step campaign, over 100 harness episodes per
condition, `A_learned` scores 35/100 and `A_oracle` 34/100, paired 5 to 4, p = 1.00
(Section 6.8).

It should also be read alongside `A_frozen`, which cannot see the scene at all and
scores 50.0 percent against `A_learned`'s 50.7. Two independent manipulations of System
2, one making it perfect and one making it blind, both leave success unchanged.

Worth being explicit about a confound the design creates here: System 1 is trained with
teacher forcing on ground truth sub-goals (Section 4.3), so at evaluation it is the
learned selector that is out of distribution, not the oracle. That direction of bias
favours the oracle. It still did not win.

### 6.4 System 2 is indecisive rather than inaccurate

During video review, the selector was observed flipping between "place the bowl" and
"close the drawer" while the manipulator hovered over an open drawer. That frame is
genuinely ambiguous for a memoryless classifier: an open drawer with the gripper above
it looks nearly identical whether the bowl is still held or was placed a moment
earlier, and the disambiguating evidence occurred many steps earlier. Because the action
queue is flushed on every sub-goal change (Section 3.7), the controller was being
redirected faster than it could complete any motion.

Hysteresis threshold sweep, 30 paired episodes on one checkpoint:

| thresholds (forward, backward) | success | forward transitions | backward transitions | agreement with oracle |
|---|---|---|---|---|
| none (1, 1) | 12/30 | 68 | 22 | 63% |
| (2, 4) | 16/30 | 70 | 19 | 64% |
| (3, 8) | 17/30 | 62 | 11 | 65% |
| (5, 12) | 16/30 | 64 | 13 | 67% |
| (3, 3) symmetric | 16/30 | 69 | 18 | 65% |
| oracle reference | 17/30 | 42 | 13 | 100% |

Two readings, and the second is the interesting one.

**The effect is robust rather than tuned.** Even the mildest setting recovers most of
the gap, and every setting tried lands between 16 and 17 out of 30. No threshold search
was required and no specific value is critical to the result, which matters because a
finding that depended on choosing exactly the right threshold would be a much weaker
claim.

**Agreement with the oracle moves only from 63 to 65 percent while backward transitions
fall by half.** The selector did not become more accurate; it became more stable. Its
instantaneous predictions were largely correct already, and the cost was paid in
committing to them too eagerly. This distinction matters for the Q3 diagnosis: the
reasoning module's weakness is not primarily a perception or classification failure.

At the full 300 episode sample the advantage disappears entirely: `A_debounced` scores
50.3 percent against `A_learned`'s 50.7, paired 50 to 49, p = 1.00. The honest claim is
that hysteresis fixes a real and visible pathology in the index trace and does not move
task success.

### 6.5 Error recovery: the mechanism fires and barely helps

Perturbations are injected at 40 percent of the episode:

| class | what it does |
|---|---|
| `none` | the control |
| `forced_drop` | force the gripper channel fully open while carrying |
| `action_noise` | add Gaussian noise (sigma 0.5) to actions for 20 steps, clipped to range |
| `object_shift` | teleport the first goal object about 5 cm in the plane |
| `visual_shift` | scale both camera images by 0.55 |

Success, 50 episodes per cell:

| class | `A_learned` | `A_frozen` | `A_debounced` |
|---|---|---|---|
| none (control) | 26/50 = 52% | 23/50 = 46% | 28/50 = 56% |
| **forced_drop** | **19/50 = 38%** | **13/50 = 26%** | 14/50 = 28% |
| action_noise | 16/50 = 32% | 17/50 = 34% | 13/50 = 26% |
| object_shift | 12/50 = 24% | 10/50 = 20% | 12/50 = 24% |
| visual_shift | 27/50 = 54% | 27/50 = 54% | 22/50 = 44% |

Backward sub-goal transitions per episode, where `A_frozen` is 0.00 by construction:

| class | `A_learned` | `A_frozen` | `A_debounced` |
|---|---|---|---|
| none | 2.14 | 0.00 | 0.32 |
| **forced_drop** | **2.80** | **0.00** | 1.40 |
| action_noise | 1.68 | 0.00 | 0.64 |
| object_shift | 2.80 | 0.00 | 0.92 |
| visual_shift | 1.02 | 0.00 | 0.36 |

**Q2.1, detection: yes, clearly.** Forcing the gripper open while the robot carries an
object raises backward transitions from 2.14 to 2.80 per episode, and teleporting the
target object raises them to 2.80 as well, while a purely cosmetic visual shift *lowers*
them to 1.02. The ordering tracks which perturbations genuinely undo progress rather
than which ones change the image. The closed loop is behaving exactly as designed: it
observes that the object is no longer held, and it revises its belief backwards.

**Q2.2, outcome: suggestive at best.** Under `forced_drop`, `A_learned` leads `A_frozen`
38 percent to 26 percent, paired 7 episodes won against 1 lost, p = 0.07. That is the
one place in this study where the closed loop looks like it is paying for itself, and it
is the perturbation class the mechanism specifically targets. But pooled over all four
perturbation classes the gap falls to 37.0 percent against 33.5 percent, paired 20 to
13, p = 0.30. Two of the four classes are exact ties or reversals.

**The within condition split removes the last alternative explanation.** Restricting to
`A_learned`'s own perturbed episodes and splitting them by whether the sub-goal actually
regressed:

```
regressed at least once    n = 97    success 13/97   (13%)
never regressed            n = 103   success 61/103  (59%)
```

Episodes where the system noticed and revised are far *less* successful than episodes
where it did not. That is not evidence that regression harms: it is confounded, because
regression is triggered by things going wrong, so the split is largely a proxy for
episode difficulty. What it does rule out is the optimistic reading. Regression
indicates a failure but does not reverse it.

**Step counts show no separation either.** Mean steps to success on the unperturbed
control: 243.9 for `A_learned`, 255.3 for `A_frozen`, 266.8 for `A_debounced`.
Conditioning changes which episodes succeed, not their duration.

One caveat carried into Section 8.2: the unperturbed controls differ across the three
conditions (52, 46 and 56 percent), so **absolute** post perturbation success is quoted
rather than each condition's own delta, which would flatter whichever condition had the
lowest control.

### 6.6 Bottleneck decomposition: agreement does not predict success

Per task, from the 30 episode harness sweep (3 episodes per task):

| task | `A_learned` success | oracle agreement | `A_oracle` success |
|---|---|---|---|
| 0 | 0/3 | 37% | 0/3 |
| 1 | 1/3 | 72% | 2/3 |
| 2 | 3/3 | 38% | 3/3 |
| 3 | 3/3 | 84% | 3/3 |
| 4 | 2/3 | 92% | 1/3 |
| 5 | 3/3 | 44% | 1/3 |
| 6 | 1/3 | 82% | 2/3 |
| 7 | 3/3 | 54% | 2/3 |
| 8 | 0/3 | 44% | 0/3 |
| 9 | 2/3 | 81% | 2/3 |

**Reasoning quality and task outcome are close to uncorrelated across the suite.** The
Pearson correlation between per task agreement and per task success is r = 0.13 over ten
tasks, which at n = 10 is nowhere near significance (the 5 percent critical value is
about 0.63). On the larger 100 episode harness of the second campaign the same
correlation is r = 0.36, still not significant.

The individual cases make the point more vividly than the correlation does. **Task 2
succeeds 3 out of 3 at 38 percent agreement**: System 2 is wrong about the stage most of
the time and the robot completes the task anyway. **Task 4 succeeds 2 out of 3 at 92
percent agreement**, and the oracle, which is right 100 percent of the time, does
*worse* on it at 1 out of 3. **Task 0 and task 8 fail for every condition including
`A_oracle`**, which places them beyond the reach of any reasoning improvement.

If reasoning were the binding constraint, agreement and outcome would move together.
They do not.

An important consequence for Q3.1: the aggregate agreement figure of 62.8 percent, which
looks poor in isolation, is not the explanation for the failures. The mistakes System 2
makes are not the ones that matter.

### 6.7 Would more training help?

This is worth recording carefully, because the first answer was wrong and the way it was
wrong is instructive.

The initial reading was "no", from two observations: the training loss had flattened,
moving only 0.006 over the final tenth of training, and the selector's label accuracy had
saturated at 0.999. Both looked like convergence.

Checkpoints are saved every 10,000 steps, so the question is answerable rather than
arguable. Evaluating three of them on the same 100 scenes (seed 1000):

| training steps | `A_learned` | `C_none` | gap |
|---|---|---|---|
| 40,000 | 40.0% | 40.0% | 0.0 |
| 70,000 | 45.0% | 44.0% | +1.0 |
| 100,000 (shipped) | 51.0% | 43.0% | +8.0 |

**Success was still climbing at the point where training stopped.** No single pairwise
comparison reaches significance at 100 episodes (40k against 100k gives p = 0.09, with
23 scenes won against 12 lost), but three points rising in order is hard to read as
noise.

The lesson: **training loss is a poor proxy for task success here.** That last 0.006 of
loss was worth roughly 6 points of success rate. A flat loss curve says the optimiser has
stopped finding easy gains on the imitation objective, not that the policy has stopped
getting better at the actual task.

Two things survive:

* **System 2 genuinely cannot improve with more steps.** It already fits its labels at
  0.999 accuracy. Its ceiling is label quality, which is a data problem.
* **The open statistical question is not a training problem.** Resolving the 4.4 point
  `A_learned` against `C_none` gap at 80 percent power needs roughly 1,600 episodes per
  condition, about 5.4 times what was run, however long the policies train.

The gap column is the interesting part and should be treated as a hypothesis, not a
finding: at 40,000 steps the two conditions are identical, and the conditioning advantage
appears only late. That would be consistent with conditioning paying off only once the
controller is good enough to act on it. It is one seed of 100 episodes, the paired test
at 100k is still p = 0.27, and the +8.0 here is larger than the +4.4 measured when all
three seeds are pooled, so part of it is this seed being favourable.

Section 6.8 puts a substantial dent in that hypothesis.

### 6.8 A second campaign at 200,000 steps

A second campaign trained the same conditions to 200,000 steps, twice the shipped budget,
and evaluated them on the same three seeds and 300 paired episodes. Its results are not
folded into the tables above, and it is reported separately because it is a different
budget. It is included here because it bears directly on Section 6.7, and leaving it out
would be reporting only the campaign that suits the narrative.

| condition | 100,000 steps | 200,000 steps | change |
|---|---|---|---|
| `C_none` | 46.3% | 47.0% | +0.7 |
| `B_static` | 44.0% | 42.3% | -1.7 |
| `A_frozen` | 50.0% | 43.3% | -6.7 |
| `A_learned` | 50.7% | 33.7% | **-17.0** |
| `A_shuffled` | 34.0% | 24.0% | -10.0 |

**The unconditioned floor is flat and the conditioned condition collapses.** `C_none`
moves 0.7 points across a doubled training budget, which is well inside noise. The
conditioned policy loses 17 points. `A_frozen`, which runs on the same weights but
ignores the selector, loses only 6.7, so a substantial part of the loss is attributable
to the live selector rather than to the controller alone.

Two things this does and does not support:

* It **does not** support "train longer". The trend visible inside the first 100,000
  steps (Section 6.7) does not continue, and for the conditioned condition it reverses
  sharply. The shipped budget of 100,000 steps is, on this evidence, at or near the best
  of the two measured budgets rather than a floor.
* It **does not** overturn any conclusion in Section 7. The oracle null reproduces at the
  larger budget and larger harness sample (35/100 against 34/100, p = 1.00), the
  positive control still separates, and the floor is unchanged. If anything it
  strengthens the execution bound diagnosis, since the condition that leans hardest on
  the conditioning channel is the one that degrades.

Honest caveats on this table. It is one training run per condition at the new budget, so
the 17 point move sits within the range a single retraining of this architecture could
plausibly produce and must not be read as finely resolved. The
training log for the 200,000 step run is not among the fetched artifacts, so the budget
is attested by the campaign marker file and the harness tag rather than by a logged step
count. And no mechanism for the collapse has been established; the selector's harness
agreement is essentially unchanged (62.2 percent at 200k against 62.8 at 100k), which
rules out the simplest explanation.

---

## 7. Discussion: answering the questions

### 7.1 The questions, one at a time

**Q1.1, does conditioning matter at all: the channel is used and the benefit is
unresolved.** A random valid sub-goal costs 16.7 points (34.0 percent against 50.7,
p below 1e-7, 69 paired wins against 19). System 1 reads the sub-goal token and its
behaviour depends on the content. The weaker form of the same question, `A_learned`
against `C_none`, gives a nominal 4.4 point advantage at p = 0.27, which this sample
cannot resolve.

The combination is informative rather than contradictory. It says the channel is a
genuine input with real influence, but that supplying the *correct* value is worth much
less than supplying an incorrect value costs. That asymmetry is itself a finding: this
System 1 is highly sensitive to being misled while having little to gain from being
helped. ACT already infers the stage from the images, so a correct token is largely
redundant, while a contradictory one overrides the camera.

**Q1.2, does online selection matter: no measurable effect.** `A_frozen`, an open loop
clock that never looks at the scene, scores 50.0 percent against `A_learned`'s 50.7
(p = 0.89), and hysteresis changes nothing either (50.3 percent, p = 1.00). Closing the
loop, which is the defining feature of a dual system architecture as opposed to a
sub-goal conditioned one, does not improve success on this benchmark with this
controller.

Why an open loop clock is competitive: the `libero_10` tasks are highly stereotyped. The
same demonstrations, similar object layouts, similar durations. A schedule spread over
the episode is approximately right most of the time, and an approximately right and
stable signal outperforms a frequently wrong and jittery one, particularly when every
sub-goal change discards the planned action chunk. **This is a property of the benchmark
rather than a general truth**, and Section 10.1 proposes the experiment that would
separate the two.

**Q1.3, is the naive baseline sufficient: it is worse than no conditioning.** `B_static`
at 44.0 percent sits below `C_none` at 46.3 and 6.7 points below `A_learned` (p = 0.06). A
constant sub-goal, which is correct at the start and stale for the rest of the episode,
misleads the controller more than providing no signal at all.

This has a methodological consequence. Because `B_static` is worse than `C_none`, the gap
between `B_static` and `A_learned` cannot be interpreted as "the value of progress
tracking" in the clean way Section 5.2 hoped. Part of that gap is `A_learned` avoiding
the harm that `B_static` causes, rather than `A_learned` adding benefit.

Read alongside `A_shuffled`, this completes a consistent picture: this System 1 is
strongly sensitive to incorrect conditioning in both the deliberately wrong and the
merely stale case.

**Q2.1, is the disturbance detected: yes, clearly.** 2.80 backward transitions per
episode under `forced_drop` against 2.14 unperturbed and 1.02 under a purely visual
disturbance, with `A_frozen` structurally at 0.00. The ordering across perturbation
classes matches which perturbations genuinely undo progress. The mechanism is real,
specific and measurable.

**Q2.2, does detection aid recovery: suggestive, not established.** Under the
perturbation the mechanism targets, `A_learned` leads the structurally non regressing
control 38 percent to 26, paired 7 to 1, p = 0.07. Pooled over all four perturbation
classes that falls to 37.0 against 33.5, p = 0.30. This is the only place in the study
where the closed loop looks like it might pay for itself, and 50 episodes per cell is not
enough to rest on. The honest summary: System 2 detects the disturbance, and it is not
clear that System 1 can exploit that detection.

**Q3.1, how accurate is System 2: 62.8 percent agreement, and accuracy is not the
constraint.** Task 2 reaches 3 out of 3 success at 38 percent agreement; task 4 reaches 2
out of 3 at 92 percent. Across the ten tasks, agreement and success correlate at r = 0.13.
The mistakes the selector makes are not the ones that determine outcomes. Section 6.4
sharpens this: hysteresis halves the backward transition rate and moves agreement by 2
points, so the selector's problem was stability rather than classification, and fixing the
stability moved success not at all.

**Q3.2, would perfect reasoning help: no.** `A_oracle` at 16/30 against `A_learned` at
18/30 on matched episodes, reproduced at 35/100 against 34/100 on a larger harness at a
different training budget. Giving System 2 perfect progress knowledge produces no
improvement. Real reasoning headroom existed (agreement was 62.8 percent, so 37 points of
it were available to close), and closing all of it produced no gain. **This is the result
that determines what to build next, and it says unambiguously that the next unit of
effort does not belong in System 2.**

### 7.2 Four independent routes to one conclusion

| route | evidence | what it rules out |
|---|---|---|
| corrupting the signal hurts | `A_shuffled` 34.0 against `A_learned` 50.7, p below 1e-7 | the channel being ignored by System 1 |
| a blind clock ties the selector | `A_frozen` 50.0 against 50.7, p = 0.89 | online selection having measurable value |
| perfect reasoning ties | `A_oracle` 16/30 against 18/30; 34/100 against 35/100 | reasoning accuracy being the bottleneck |
| stability fix ties | `A_debounced` 50.3 against 50.7, p = 1.00 | selector instability being the bottleneck |
| recovery fires and barely helps | 2.80 backward transitions against 0.00, 38% against 26% | the recovery mechanism being absent |

A single null result is weak evidence. It is always possible that the experiment was
underpowered, the implementation was broken, or the measurement was insensitive. What
converts these nulls into a finding is that they come from **five different
manipulations, measured in three different ways**, and that one of the five is a large,
unambiguous positive.

The first row is therefore essential rather than incidental. It establishes that the
instrument has **sensitivity**: this protocol, at this sample size, on this System 1, can
detect a 16.7 point difference when one exists. Without it, the nulls would be consistent
with an experiment that simply cannot see anything. With it, they are evidence that the
effects being sought are genuinely small or absent.

### 7.3 What it means together

Every route leads to the same place: **the system is execution bound**. System 1's
capacity to carry out a correctly identified sub-goal is the binding constraint, and the
reasoning module cannot help because it is not what is failing.

This is worth stating carefully, because it is easy to over generalise. The finding is
**not** "dual system architectures do not work". It is that on this benchmark, at this
data scale (379 demonstrations), with this controller (ACT at 51.6M parameters), the
reasoning half has no headroom to exploit. A dual system architecture assumes that
reasoning is the bottleneck. That assumption was tested directly here, by substituting a
perfect reasoner, and it was not supported.

**Why the recovery result is the most useful negative.** It is common to argue that a
closed loop system "would" recover from disturbances, and the argument is usually left at
the level of architecture. Here the mechanism is instrumented, it demonstrably activates
(2.80 backward transitions per episode against 0.00 for the structurally incapable
control), and it produces at most a marginal benefit. That is a far more specific
statement than "recovery did not improve success". It localises the failure: the
detection half of error recovery works and the execution half does not, so improving the
detector is of little value and improving the controller is of the greatest value.

**And a caveat that cuts across all of it.** Training variance is not captured in any
of the intervals here, since each trained condition has a single seed. The conclusions
above rest on the *pattern*
across five manipulations and three measurement paths, not on any single gap, and that is
the only way they can be supported at this training budget.

---

## 8. Limitations

Grouped by whether they limit the system or the evaluation, because they have different
implications for what to trust.

### 8.1 Limitations of the system design

1. **System 1 is language blind.** ACT receives no instruction text, so the only route by
   which task structure reaches the controller is the sub-goal index, and the task
   embedding hard codes exactly ten tasks. The policy cannot generalise to an unseen
    instruction or an eleventh task. This was a deliberate choice (Section 3.2) that
    produced a clean ablation at the cost of generality.
2. **The interface bandwidth is about 4.7 bits per step.** An integer from a 26 element
   vocabulary cannot express object pose, spatial relations, or partial progress within a
   stage. If the execution ceiling were lifted, this would likely become the next binding
   constraint.
3. **Conditioning is additive rather than multiplicative.** FiLM was the identified
   alternative and was not tried, because the check for whether it was needed
   (`A_shuffled`) passed.
4. **System 2 is memoryless.** It classifies from a single frame plus proprioception and
   task identity, and Section 6.4 documents the resulting ambiguity at the drawer.
   Hysteresis mitigates the symptom without addressing the cause.
5. **`C_none` is not literally stock ACT.** It carries the unused conditioning pathway
   (Section 4.2). This is conservative for the comparison but should be stated.
6. **The sub-goal decomposition is rule based** and therefore does not transfer to non
   templated instructions. The obvious alternative, relabelling with a frozen VLM, was
   attempted and failed at 24.3 percent agreement, at chance for a four phase task
   (Section 4.4).
7. **The skill vocabulary is fixed** at 25 skills derived from ten known instructions.
   Nothing here generalises to an instruction the splitter has not seen.
8. **Training uses teacher forcing**, so the learned selector is out of distribution at
   evaluation in a way the oracle is not. This biases in favour of `A_oracle`, which
   still did not win, so it does not threaten the Q3.2 conclusion. It would matter for
   any positive result.

### 8.2 Limitations of the evaluation

9. **One training seed per condition.** The three evaluation seeds vary which episodes are
   run, not how the model was initialised, so none of the intervals here include training
    variance, and every trained condition is a single run whose gap could in principle be
    a retraining artefact. **This is the single largest caveat in the study.**
10. **Sample size.** 300 episodes per condition gives roughly plus or minus 5.6 points at
    95 percent confidence. `A_learned`, `A_frozen`, `A_debounced` and `C_none` all sit
    inside that band. Resolving the main comparison at 80 percent power needs about 1,600
    episodes per condition.
11. **`A_oracle` is measured on 30 episodes, not 300**, because it cannot run through the
    standard evaluator. Its interval is roughly plus or minus 17 points, so "perfect
    sub-goals do not help" is a weaker statement than the 300 episode rows and should be
    read together with the `A_frozen` result. The 100 episode replication at the second
    budget partly addresses this.
12. **Two measurement paths.** The same checkpoint scores 50.7 percent through
    `lerobot-eval` over 300 episodes and 60.0 percent through the harness over 30.
    Comparisons are made only within one path, never across.
13. **The oracle is coarse.** Roughly two BDDL predicates per task and nothing for
    intermediate states, so "no reasoning headroom" means none detectable with this
    oracle. A finer oracle, one that knew "the bowl is in the gripper", might reveal
    headroom this one cannot see.
14. **Label quality.** 24 percent of episodes have a segment count disagreeing with their
    sub-goal count, concentrated on tasks 0, 1, 6, 7 and 8, which are largely the tasks
    the selector performs worst on. Cause and effect are not separated: the selector may
    be poor on those tasks because the labels are poor, or the labels may be poor because
    those tasks are genuinely harder to segment. The labels are known to be wrong on
    task 2.
15. **`A_frozen`'s schedule is spread over the 520 step cap** while successful episodes
    finish in roughly 200 to 470 steps, so part of its competitiveness may come from a
    schedule that happens to suit these tasks rather than from open loop control being
    inherently better.
16. **Perturbation controls differ across conditions** (52, 46 and 56 percent), so
    absolute post perturbation success is quoted rather than each condition's own delta,
    which would flatter whichever condition had the lowest control.
17. **Success rate is a slow concentrating metric.** It is one Bernoulli trial per
    episode. The backward transition counts are a better instrumented signal and are
    where the recovery conclusion actually rests.
18. **Benchmark specificity.** Stereotyped tasks favour open loop schedules, so the Q1.2
    conclusion in particular may not transfer to a setting with variable layouts or
    durations.
19. **The 200,000 step campaign is reported but not explained.** A 17 point collapse in
    the conditioned condition with a flat floor is a real observation with no established
    mechanism (Section 6.8).

---

## 9. Lessons learned

### 9.1 Measurement hazards

Several defects during this campaign each produced a plausible wrong number rather than
an error, and the pattern generalises to any simulation pipeline.

* `eval.batch_size=1` reported 0.0 percent success for every policy, including stock ACT,
  while running rollouts to completion and writing normal looking videos.
* `n_action_steps` was left at the checkpoint default of 100 in the perturbation harness,
  silently disabling the closed loop the harness existed to measure. Its unperturbed
  control read 30 percent instead of 45.
* The scene was not settled after reset in that same harness, depressing its control by
  15 points.
* A camera orientation flip fed the policy a mirrored world. LIBERO renders bottom up and
  the flip has to be applied before encoding.
* Camera naming differs between the environment and the dataset (`agentview_image` against
  `observation.images.image`). Normalization statistics are keyed by name, so a mismatch
  corrupts training silently.

**None raised an exception.** Each was found by a **control disagreeing with itself**: the
same condition, measured two ways, giving two answers. That is the practical reason the
unperturbed control is reported next to every perturbed number in Section 6.5, and the
reason `A_learned` appears in both the evaluator table and the harness table rather than
being quoted once. In a pipeline of this kind, a silent plausible value is considerably
harder to detect than a crash.

Two further traps specific to the plugin, both silent, both documented in Section 4.1: a
hyphenated distribution name disables policy discovery, and a standalone script that does
not call `register_third_party_plugins()` cannot load its own checkpoints.

### 9.2 What the analysis got wrong and had to correct

Recorded because the corrections are more instructive than the original claims.

* **"More training would not help."** Argued from a flat loss curve and a saturated
  selector accuracy. Both were true and neither was relevant; the last 0.006 of loss was
  worth about 6 points of success (Section 6.7). The correction was then itself
  complicated by the 200,000 step campaign (Section 6.8), so the current position is that
  the relationship between budget and success is not monotone and is not established.
* **"A predicate count is a usable oracle."** Rejected on analysis before it could produce
  a number, because it lags the training labels by a phase and cannot reach the top index
  on six of ten tasks (Section 5.4). It would have inverted the answer to the study's most
  important question.
* **"Row numbers are fine."** They are not Markovian and they fragment the training
  signal (Section 3.5). The fix turned out to be null on success, which is the honest
  result, and it opened a held out task experiment that pure row numbering forecloses.

---

## 10. Future work

### 10.1 What would be done next, ordered by expected value

1. **Multiple training seeds, before anything else.** This is the cheapest fix to the
   largest limitation, and training variance would place every non-shuffled gap inside
   the range a single retraining could produce. Three seeds per trained
   condition is nine training runs, roughly 36 GPU hours, and it would convert "these four
   conditions are indistinguishable" from an artefact of sample size into a statement
   about the architecture.
2. **More evaluation episodes.** Resolving the `A_learned` against `C_none` comparison at
   80 percent power needs about 1,600 episodes per condition. It needs no retraining and
   is the only thing that turns the central "unresolved" into an answer.
3. **Invest the next unit of effort in System 1, not System 2.** Every result points here.
   A diffusion or flow matching action head, a stronger visual backbone, or more data
   would raise the ceiling that currently makes the reasoning half irrelevant. The most
   informative single experiment available is to re run this exact ablation against a
   stronger System 1, because the ablation is now a validated instrument with known
   sensitivity: it detected 16.7 points when 16.7 points existed.
4. **Explain the 200,000 step collapse.** A conditioned condition losing 17 points while
   the unconditioned floor holds flat is either a real overtraining pathology specific to
   the conditioning pathway or an artefact of one training run. Which of those it is
   changes the recommended training budget.
5. **Zero shot execution on a held out task**, which the skill vocabulary `v25` makes
   possible. Hold out one task whose sub-goals all appear elsewhere in the `v25`
   vocabulary, train on the other nine, evaluate on the held out one. Because the rows
   are shared across tasks, the held out task's rows are trained, by other tasks, and the
   question becomes real: does a sub-goal channel carry a transferable skill, or only a
   position in a script? This is the strongest argument for the vocabulary change.
6. **A benchmark that can distinguish open loop from closed loop.** Randomised object
   placement, variable task duration, or mid episode goal changes would break the
   stereotypy that makes `A_frozen` competitive. As it stands, Q1.2 was answered on a
   benchmark somewhat unsuited to asking it.
7. **A finer oracle**, covering intermediate states such as "object grasped", to test
   whether the Q3.2 null is a property of the system or of the coarse oracle.
8. **Label quality audit** on tasks 0, 1, 6, 7 and 8, and a fix for task 2 using BDDL
   predicates where they exist rather than the gripper alone, before adding any selector
   capacity.
9. **Re run the forced drop comparison at a larger sample.** 7 wins against 1 at p = 0.07
   on 50 episodes is the only sign in the study that the closed loop pays for itself.
10. **FiLM conditioning**, which becomes worth testing as soon as the execution ceiling
    lifts.
11. **Memory in System 2**, but only if reasoning headroom reappears. The drawer ambiguity
    is real and memory would address it directly. It carries a specific trap worth
    recording: phases persist for 50 to 100 consecutive frames, so a "previous phase" input
    trained with teacher forcing has a degenerate solution of copying its own input and
    ignoring the image, which would score well in training and collapse to a constant at
    rollout. Any memory feature must be corrupted during training, with dropout and
    scheduled sampling. A truly recurrent selector is additionally blocked by LeRobot
    sampling random frames rather than contiguous sequences, so it would require dataloader
    changes affecting every condition.
12. **Measure step count as a first class metric**, not just success. A policy that
    succeeds faster is better and success rate hides that entirely.

### 10.2 Moving to a Helix or GR00T style architecture

The system built here and the current generation of production dual system VLAs (Figure's
Helix, NVIDIA's GR00T N1) share a shape and differ in almost every detail. Setting out the
differences concretely is useful because it says what would have to change and what it
would be expected to buy.

| aspect | this system | Helix / GR00T N1 style |
|---|---|---|
| System 2 | 162k parameter MLP over pooled ACT features | pretrained Vision Language Model, roughly 2B to 7B parameters |
| Interface | one discrete integer, about 4.7 bits per step | continuous latent vector or token sequence, hundreds of dimensions |
| Coupling | additive offset on one encoder token | cross attention from System 1 into System 2's token stream |
| System 1 | ACT, 51.6M parameters, single forward pass | diffusion or flow matching transformer action head |
| Training | System 2 supervised on derived labels, System 1 on demonstrations with teacher forcing | end to end, jointly, on far larger and more diverse data |
| Rates | both at 10 Hz, System 2 recomputed every step | System 2 at roughly 7 to 9 Hz, System 1 at roughly 200 Hz, decoupled |
| Language | System 1 is language blind | instruction consumed by the VLM, so the system generalises to new phrasing |

**What it would take**, in rough order of effort:

1. Replace the selector with a pretrained VLM producing token embeddings rather than a
   phase classification. Frozen VLM plus a trained adapter is the affordable version; full
   fine tuning is not at this compute budget.
2. Replace additive conditioning with cross attention, so System 1 can attend to the parts
   of the semantic representation relevant to the current motion rather than receiving one
   summed offset.
3. Replace ACT with a diffusion or flow matching head, both for multimodality and because
   that is what the published systems use, which makes the comparison meaningful.
4. Decouple the rates, with System 2 running asynchronously and its latent cached between
   updates, which is what makes a large System 2 affordable in a real control loop.

**What it could achieve.** The specific failure observed in this study is instructive.
System 2 flipped between "place the bowl" and "close the drawer" because a single frame
plus a 4.7 bit output cannot represent "I am holding the bowl". A VLM emitting a
continuous latent does not need explicit memory to fix this: it sees the whole scene and
can encode "the bowl is in the gripper, the drawer is open" directly, because the
representation has room for it. The higher bandwidth interface addresses the root cause
that hysteresis only masked. Beyond that, such a system would generalise to unseen
instructions and objects, which this one structurally cannot.

**What it would cost, and one thing it would lose.** The compute requirement is orders of
magnitude larger, and the data requirement is the real barrier: these systems are trained
on human video, simulation, and real robot data at a scale a benchmark subset cannot
approach. The call volume alone is worth quantifying: at ten conditions, ten tasks, ten
episodes and 520 steps, one evaluation sweep is roughly 364,000 System 2 invocations. A
frozen large model at that volume changes the cost of the ablation, not only the cost of
the policy.

The thing that would be lost deserves emphasis, because it is not usually mentioned.
**This ablation would no longer be possible.** The discrete, legible interface is exactly
what made `A_shuffled`, `A_frozen` and `A_oracle` constructible, and those three
conditions are what turned a set of success rates into a diagnosis. With a continuous
latent there is no principled "deliberately wrong but plausible" value, no way to express
ground truth progress as a latent without inverting the model, and no way to draw System
2's belief on a video frame and compare it to reality. A higher bandwidth interface buys
capability and costs interpretability, and a project whose deliverable is an ablation
study should weigh that trade explicitly rather than assume more bandwidth is strictly
better.

**The honest caveat on expected gains.** This study found the system to be execution
bound. A larger System 2 alone would therefore not have helped here. The gain from a
Helix or GR00T style redesign would have to come jointly from the stronger action head,
the higher bandwidth interface, and the larger data scale, not from the better reasoner,
and that ordering is the direct consequence of the `A_oracle` result.

---

## Appendix A: where every number comes from

Every table above is recomputable from the shipped artifacts. Paths are relative to the
artifacts root for the 100,000 step campaign unless stated otherwise.

### A.1 Main ablation (Section 6.1, 6.2)

* Per episode outcomes: `eval/<condition>_seed{1000,2000,3000}/eval_info.json`, field
  `per_task[i].metrics.successes`.
* Pooled rates and Wilson intervals: `analysis/results.csv`.
* Paired McNemar against the floor: `analysis/comparisons.csv`. The p values against
  `A_learned` quoted in Section 6.1 are recomputed by pairing on
  `(seed, task_id, episode)`.
* Figure: `report/fig1_ablation.png`.

### A.2 Harness and oracle (Section 6.3, 6.6)

* Per episode records: `videos/learned_v25/*.json` and `videos/oracle_v25/*.json`, 30
  episodes each, keyed `(task_id, episode)`. Fields used: `success`,
  `s2_oracle_agreement`, `n_forward_transitions`, `n_backward_transitions`.
* Second campaign, 100 episodes per condition: `videos/<condition>_200000_full/*.json`
  under the 200,000 step artifacts root, summarised by `scripts/16_harness_summary.py`
  into `analysis/harness_results.csv`.

### A.3 Perturbations (Section 6.5)

* `perturbation/<condition>/{none,forced_drop,action_noise,object_shift,visual_shift}.jsonl`,
  one JSON record per episode, 50 per file. Fields used: `success`,
  `n_backward_transitions`, `n_forward_transitions`, `steps_to_success`.
* Conditions map: `A_learned` is `perturbation/A_merged25/`, `A_frozen` is
  `perturbation/A_frozen25/`, `A_debounced` is `perturbation/A_debounced/`.
* Figure: `report/fig3_recovery.png`.

### A.4 Training (Section 4.2, 5.9)

* `<condition>.train.log`. Parameter counts from the `num_learnable_params` line, wall
  clock and final losses from the final `step:100K` line.

### A.5 The decomposition and its vocabularies (Section 3.3, 3.5)

`subgoals.json` carries, per task, the sub-goal descriptions and a `canonical` map from
vocabulary name to class ids aligned index for index:

| task | instruction | phases | `v25` classes |
|---|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 4 | 0, 1, 2, 3 |
| 1 | put both the cream cheese box and the butter in the basket | 4 | 4, 5, 6, 7 |
| 2 | turn on the stove and put the moka pot on it | 3 | 8, 9, 10 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 3 | 11, 12, 13 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 4 | 14, 15, 16, 17 |
| 5 | pick up the book and place it in the back compartment of the caddy | 2 | 18, 19 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 4 | **14**, 20, 21, 22 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 4 | **0, 1, 4, 5** |
| 8 | put both moka pots on the stove | 4 | **9, 10, 9, 10** |
| 9 | put the yellow and white mug in the microwave and close it | 3 | **16**, 23, 24 |

Bold entries are classes shared with an earlier task, which is the merging that `v25`
performs. Task 8's `[9, 10, 9, 10]` is the sharpest case: the two moka pots are one
skill, and only the image distinguishes them.

---

## Appendix B: condition name map

The report uses the public condition names. The artifacts use the internal names from the
campaign scripts. This table is the mapping, and it also records which checkpoint and
which vocabulary each condition runs on.

| report name | artifact name | checkpoint | vocabulary | evaluation mode | hysteresis |
|---|---|---|---|---|---|
| `A_learned` | `A_merged25` | `A_merged25` | v25 | learned | (1, 1) |
| `C_none` | `C_none` | `C_none` | n/a | none | (1, 1) |
| `B_static` | `B_static25` | `B_static25` | v25 | static | (1, 1) |
| `A_shuffled` | `A_merged25_shuffled` | `A_merged25` | v25 | shuffled | (1, 1) |
| `A_frozen` | `A_frozen25` | `A_merged25` | v25 | frozen | (1, 1) |
| `A_oracle` | `A_oracle25` / `oracle_v25` | `A_merged25` | v25 | oracle | (1, 1) |
| `A_debounced` | `A_debounced` | `A_learned` | v25 | learned | (3, 8) |

One entry carries a caveat.

**`A_oracle` cannot run through `lerobot-eval` at all** and appears only in harness
tables. See Section 5.8.


---

## Appendix C: reproduction commands

```bash
# 0. environment check
python scripts/00_check_setup.py

# 1. the decomposition and the per-frame labels
python scripts/01_make_subgoals.py
python scripts/02_label_dataset.py

# 2. train the three conditions that need training, about 4 hours each
python scripts/03_train.py --condition A_learned
python scripts/03_train.py --condition B_static
python scripts/03_train.py --condition C_none

# 3. the ablation: 3 seeds x 100 episodes for every condition
python scripts/05_sweep.py
python scripts/07_analyze.py

# 4. error recovery and the rollout videos with System 2 overlaid
python scripts/06_perturb.py --condition A_learned
python scripts/06_perturb.py --condition A_frozen
python scripts/08_video.py --condition A_learned
python scripts/08_video.py --condition A_oracle
```

Two environment facts that will otherwise cost time:

```bash
export MUJOCO_GL=egl     # binds at import time; setting it inside Python does nothing
```

and `import libero` prompts interactively without `~/.libero/config.yaml`, which hangs a
non interactive run. The setup script writes it first.

---

## Appendix D: corrections against the compact report

The compact three page report and this document describe the same campaign. Where a
figure here differs from the compact version, the value here was recomputed from the
artifacts and the difference is recorded below rather than silently applied.

| item | compact report | this report | source |
|---|---|---|---|
| System 2 parameter count | "approximately 2M parameters" | **161,988** | `num_learnable_params` in the training logs, and the module definition in `modeling_subgoal_act.py` |
| Selector loss weight | "weighted at 0.1" | **0.5** | `selector_loss_weight` in the logged training config |
| Bottleneck comparison | "`A_learned` and `A_oracle` both succeed in 47% of episodes ... 100 episodes per condition" | **18/30 = 60.0% against 16/30 = 53.3%**, and 35/100 against 34/100 in the second campaign | `videos/learned_v25/`, `videos/oracle_v25/`, `analysis/harness_results.csv` |
| Shuffled effect size | "costs 16.7 points" | unchanged, 50.7 against 34.0 | `analysis/results.csv` |
| Gripper thresholds | "open above 0.65, closed below 0.35 of maximum gripper opening" | thresholds are **per episode**, at 0.35 and 0.65 of that episode's own open to close span, with the open reference floored at 0.080 | `scripts/02_label_dataset.py` |
| Interface bandwidth | not stated | about **4.7 bits per step** under `v25` (26 classes) | arithmetic |

The direction of every correction is neutral or unfavourable to the system, and none of
them changes an answer in Section 7.

Two further notes on the compact report's assets. Its three figures were regenerated
from this campaign's numbers; the current figures are in the artifacts under `report/`.
