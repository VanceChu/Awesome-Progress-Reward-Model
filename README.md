# Awesome-Progress-Reward-Model

Robotic reward models are becoming a central interface for long-horizon manipulation, test-time adaptation, failure recovery, and planner-guided control. Their importance comes from learning more than a generic scalar reward: some methods estimate global task completion, some compare short-horizon forward-versus-backward transitions, and newer systems combine dense process rewards with preference, success, or verifier signals. This repository summarizes representative research on robotic reward modeling around the `progress vs delta` question, clarifies the main design landscape, and highlights the broader system interfaces that make these signals useful in practice.

**🤝 Contributions are welcome! Please feel free to submit an issue to add papers or suggest taxonomy changes.**

If you find this repository useful, please consider giving it a star ⭐. Feel free to share it with others.

Contemporary robotic reward models can be broadly organized into three semantic families:

- 📈 `absolute progress`: estimate where the agent currently is on the task
- ↔️ `relative progress`: estimate how much progress or regress happened from one state to another
- 🧩 `hybrid reward systems`: combine process reward with preference, success, or verifier-style signals

In practice, strong systems also need a fourth perspective:

- ⚙️ `system interface papers`: how a learned reward-like signal is actually turned into RL shaping, termination, rewind, test-time verification, or planning-time ranking

This README organizes each method around five questions:

- 🎯 What does the reward head actually predict?
- 🏷️ What supervision is available?
- ⚠️ What failure modes does the method address or ignore?
- 🔌 How is the learned signal consumed downstream?
- 🤖 What role could this method realistically play inside an embodied AI project?

## 🗺️ Quick Navigation

- 🔎 [At a Glance](#at-a-glance)
- 📈 [Reward Models with Absolute Progress Prediction](#reward-models-with-absolute-progress-prediction)
- ↔️ [Reward Models with Relative Progress Prediction](#reward-models-with-relative-progress-prediction)
- 🧩 [Hybrid Reward Systems with Preference, Success, and Verifier Signals](#hybrid-reward-systems-with-preference-success-and-verifier-signals)
- ⚙️ [Reward Models as System Interfaces for RL, Termination, Rewind, Verification, and Planning](#reward-models-as-system-interfaces-for-rl-termination-rewind-and-planning)
- 🧱 [Supporting Papers for Stage-Aware and Temporal Feedback](#supporting-papers-for-stage-aware-and-temporal-feedback)
- 📌 [Repository Scope](#repository-scope)



<a id="at-a-glance"></a>
## 🔎 At a Glance

| Category | Core question | Typical supervision | Main downstream role | Main risk |
| --- | --- | --- | --- | --- |
| `Absolute progress` | Where am I on the task? | normalized time, stage labels, completion belief, future-conditioned value | monitoring, shaping, thresholding, stage control | semantic drift when time and task structure diverge |
| `Relative progress` | From A to B, did I move forward or backward? | pairwise forward/reverse pairs, signed hop labels | dense shaping, transition-level advancement scoring | pairwise correctness may not yield a stable global completion variable |
| `Hybrid reward systems` | How should progress, preference, success, and verifier signals work together? | mixed process labels, real failures, preference data, outcome checks, multi-dimensional world-model preferences | scalable reward learning, boundary modeling, evaluation | higher modeling and calibration complexity |
| `System interfaces` | How does reward prediction become useful inside a larger control loop? | inherited reward signals, online environment feedback, inverse-dynamics or verifier signals | RL, termination, rewind, verification, planning, ranking, world-model alignment | easy to confuse head semantics with system use |


<a id="reward-models-with-absolute-progress-prediction"></a>
# **📈 Reward Models with Absolute Progress Prediction**

This category contains papers that try to learn a globally interpretable completion variable. The central object is usually a scalar or structured progress estimate answering a question like: "how far along is this trajectory or state in the task?" These methods are attractive when the downstream system needs a stable notion of completion for shaping, thresholding, stage control, or monitoring.

The core inclusion rule for this section is simple: the paper must primarily treat reward learning as `global task progress estimation`, rather than as pairwise comparison or a mixed reward stack. The main strengths of this family are interpretability and easy integration into downstream control logic. Its main weakness is that supervision can become semantically noisy when normalized time, demonstration speed, or stage order do not align with real task advancement.

Within this category, an important subdivision is already visible. `ReWiND` represents the most direct route: map successful demonstrations into a monotonic completion target, then use rewind-style augmentation to inject failure-like structure. `SARM` is a correction to that route: it accepts the absolute-progress premise, but argues that completion should often be stage-aware rather than uniformly time-normalized. `TOPReward` pushes in a different direction again by asking whether a pretrained model's completion belief can stand in for an explicitly trained reward head. `ProgAgent` treats progress as a learned state-potential for continual RL. `SOLE-R1` frames progress estimation as a reasoning-heavy reward problem designed to remain useful during online RL. `ViVa` adds a foresight-oriented route by estimating scalar state value through future proprioceptive prediction inside a video-generative model.

So even inside `absolute progress`, the real comparison is not just "do they all learn completion?" but:

- how completion is defined,
- how much task structure is injected by hand,
- whether calibration comes from explicit reward training or from a zero-shot probe,
- whether robustness comes from direct regression, latent probing, state-potential learning, future-conditioned value prediction, or reasoning-oriented reward post-training,
- and whether the output is reliable enough to support boundaries such as termination or relabeling.

| Method | Title | Venue | Date | Code/Project | High-level labels | Key feature/finding |
| --- | --- | --- | --- | --- | --- | --- |
| [ReWiND](https://arxiv.org/abs/2505.10911) | ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations | CoRL 2025 (Oral) | 5/2025 | [Project](https://rewind-reward.github.io/) / [Code](https://github.com/rewind-reward/ReWiND) | `Core · Absolute Progress · Success + Rewind · Shaping + Relabel` | Direct frame-level progress regression with rewind augmentation for reward relabeling and policy improvement. |
| [SARM](https://arxiv.org/abs/2509.25358) | SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation | ICLR 2026 (Poster) | 9/2025 | - | `Core · Absolute Progress · Stage-Structured · Shaping + Termination` | Stage-aware absolute-progress modeling instead of naive normalized-time supervision. |
| [TOPReward](https://arxiv.org/abs/2602.19313) | TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics | Arxiv | 2/2026 | - | `Core · Absolute Progress · Zero-Shot Completion · Shaping + Eval` | Zero-shot completion belief extracted from token probabilities rather than a separately trained reward head. |
| [ProgAgent](https://arxiv.org/abs/2603.07784) | ProgAgent: A Continual RL Agent with Progress-Aware Rewards | Arxiv | 3/2026 | - | `Core · Absolute Progress · State Potential · Continual RL` | Progress-aware dense rewards derived from unlabeled expert videos, with OOD regularization for continual robot learning. |
| [SOLE-R1](https://arxiv.org/abs/2603.28730) | SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning | Arxiv | 3/2026 | - | `Core · Absolute Progress · Reasoning Rewarder · Online RL` | Video-language reasoning model predicts dense task progress robust enough to act as the sole reward source for online RL. |
| [ViVa](https://arxiv.org/abs/2604.08168) | ViVa: A Video-Generative Value Model for Robot Reinforcement Learning | Arxiv | 4/2026 | [Code](https://github.com/GigaAI-research/ViVa) | `Core · Absolute Progress · Foresight Value · RL Value Shaping` | Video-generative model adapted into a scalar value estimator through future proprioception prediction and value supervision. |


### Expanded Notes

- `ReWiND`: Canonical `absolute_progress` paper. Directly regresses frame-level task progress, then augments successful trajectories with rewind data so the learned progress function can support reward relabeling and policy improvement without new demonstrations.
- `SARM`: Refines the absolute-progress route by replacing naive normalized-time supervision with stage-aware progress structure. Important because it shows that `absolute_progress` does not have to mean raw `t/T` labeling.
- `TOPReward`: Pushes the same semantic direction into a zero-shot setting. Instead of separately training a reward head, it extracts completion belief from token probabilities, making progress estimation possible through a pretrained model's latent completion signal.
- `ProgAgent`: Treats progress as a learned state-potential for continual RL. It is useful because it derives dense shaped rewards from unlabeled expert videos while adding adversarial push-back style regularization against overconfident OOD scoring during online exploration.
- `SOLE-R1`: Shows that `absolute_progress` can also be framed as a reasoning-centric reward problem. It predicts dense progress from video-language evidence and is explicitly optimized for robustness when that signal is used as the only reward in online RL.
- `ViVa`: Extends absolute progress into foresight-based value modeling. Instead of only judging the current frame, it adapts a video-generative model to predict future proprioception and scalar state value, making progress estimation depend on imagined embodiment dynamics.

These papers matter because they define the cleanest route to a reward head that behaves like an explicit completion meter. If a system needs an easily inspected `how far along am I?` variable, this is usually the first category to examine.

At the same time, this category exposes the main fragility of progress modeling: if task execution is non-monotonic, multi-stage, or highly variable in pace, then a globally ordered completion label can become much noisier than it first appears. That is exactly why later papers move toward stage-aware structure, hybrid reward designs, or pairwise progress formulations.

Practically, this category is strongest when the task has an interpretable global axis such as staged assembly, insertion, or approach-and-complete manipulation. It is weaker when tasks contain loops, retries, temporary regressions, or multiple valid solution paths, because a single monotone completion signal can over-penalize productive detours or over-reward visually smooth but semantically unhelpful motion.


<a id="reward-models-with-relative-progress-prediction"></a>
# **↔️ Reward Models with Relative Progress Prediction**

This category contains papers that model `transition-level advancement` rather than `global completion`. Instead of asking where the agent is on the full task, these methods ask whether a transition, state pair, or hop moved the system forward or backward. This usually makes supervision easier to construct because pairwise ordering, forward/reverse data, and explicit regress examples are often more available than exact per-step completion labels.

The key inclusion rule here is that the primary learned object is a `relative` or `pairwise` notion of progress. These methods are especially attractive when the downstream reward is used as dense shaping, because pairwise comparisons naturally translate into per-step feedback. Their main limitation is that pairwise correctness does not automatically produce a stable episode-level completion variable unless extra machinery is added.

An important subdivision in this category is between `pairwise ordering`, `signed stepwise dynamics`, `advantage-style relative supervision`, and `analytical-reward distillation`. `VLAC` is the clean pairwise reference point: it asks whether one state or transition is better than another under task conditioning. `Robo-Dopamine` goes further by explicitly modeling progress and regress magnitudes, then attempting to fuse those pairwise judgments back into a more globally usable signal. `ARM` reframes the problem again by asking whether short temporal windows reflect forward, backward, or stagnant task advantage. `Rewarding DINO` adds a fourth pattern: it starts from analytical dense rewards in simulation, freezes strong pretrained encoders, and learns a compact visual reward predictor through pairwise ranking. That makes it a useful bridge between classical analytic rewards and learned image-based reward models.

This means the core question inside the category is not only whether supervision is stepwise, but also what kind of stepwise object is learned:

- an ordering,
- a signed stepwise delta,
- a distilled image-based proxy for analytical dense rewards,
- or a stepwise score that can later be accumulated or transformed into a potential-like reward.

| Method | Title | Venue | Date | Code/Project | High-level labels | Key feature/finding |
| --- | --- | --- | --- | --- | --- | --- |
| [VLAC](https://arxiv.org/abs/2509.15937) | A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning | Arxiv | 9/2025 | [Project](https://github.com/InternRobotics/VLAC) | `Core · Relative Progress · Pairwise · Shaping` | Pairwise vision-language-action critic for transition-level advancement scoring from forward/reverse and mismatch supervision. |
| [Robo-Dopamine](https://arxiv.org/abs/2512.23703) | Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation | Arxiv | 12/2025 | [Project](https://github.com/FlagOpen/Robo-Dopamine) | `Core · Relative Progress · Signed Progress · Shaping + Potential` | Signed progress/regress hops fused into a bounded potential-like reward signal. |
| [Rewarding DINO](https://arxiv.org/abs/2603.16978) | Rewarding DINO: Predicting Dense Rewards with Vision Foundation Models | Arxiv | 3/2026 | - | `Core · Relative Progress · Analytical Dense Rewards · Shaping + Potential` | Dense visual reward prediction distilled from analytical simulator rewards with frozen foundation encoders. |
| [ARM](https://arxiv.org/abs/2604.03037) | ARM: Advantage Reward Modeling for Long-Horizon Manipulation | Arxiv | 4/2026 | - | `Core · Relative Progress · Advantage Labels · Shaping + Reweighting` | Tri-state advantage supervision predicts forward, backward, or stagnant progress windows for long-horizon manipulation. |

### Expanded Notes

- `VLAC`: Canonical `relative_progress` reference. Builds a pairwise vision-language-action critic that predicts advancement between states, making forward/reverse pairs and goal-mismatch pairs natural supervision sources for dense reward shaping.
- `Robo-Dopamine`: Extends the pairwise route by learning signed progress and regress hops, then fusing them into a bounded potential-like signal. Important because it shows one concrete path from pairwise delta supervision back to a more global reward variable.
- `Rewarding DINO`: Trains a dense visual reward model from analytical simulator rewards rather than human preference data. It freezes DINOv3 and MiniLM, learns a small FiLM-conditioned reward head with pairwise logistic ranking, and trains on Meta-World+ plus a simulated pick-cube setup using mixed expert/random trajectories, reward-and-Cartesian binning, and two-view image observations.
- `ARM`: Replaces absolute completion labels with relative advantage supervision over short temporal windows. Important because it makes forward/backward/stagnant judgments a first-class reward target rather than only a helper signal.

This family is usually the best place to start when `exact progress labels` are difficult to define, but `pairwise directionality` is still accessible. It also handles regress cases more naturally than many absolute-progress formulations, because backward movement can be labeled directly rather than inferred indirectly.

The tradeoff is structural: a model can be excellent at saying that `B is better than A` and still be poor at telling you how close either state is to task completion. That gap between pairwise order and global calibration is the main reason this section should always be read together with the hybrid and system-interface sections.

Practically, this category is strongest when you can build lots of stepwise supervision cheaply, for example from forward/reverse rollouts, pairwise preferences over short windows, or synthetic perturbations that create obvious regress examples. It is much less comfortable when the system also needs a robust notion of `done`, because pairwise progress does not by itself define a stable completion boundary.


<a id="hybrid-reward-systems-with-preference-success-and-verifier-signals"></a>
# **🧩 Hybrid Reward Systems with Preference, Success, and Verifier Signals**

This category contains papers that move beyond the pure `absolute progress` versus `relative progress` split. They matter because real robotic systems often need more than a single dense process-reward head. Once failures, suboptimal trajectories, ambiguous outcomes, and deployment-time verification become important, the reward stack often expands to include preference ranking, success detection, or explicit verifier modules.

The inclusion rule for this section is that reward modeling is no longer treated as one scalar semantic question. Instead, the paper explicitly separates or combines multiple reward-related functions such as progress estimation, success boundaries, inter-trajectory preference, or evaluator logic. This is where the field becomes more realistic and more difficult.

There are two important sub-directions inside this category. One is the `hybrid dense reward` route, represented by papers such as `Adapt2Reward`, `RoboReward`, `Robometer`, `RoboTracer`, `Large Reward Models`, `Generalizable Dense Reward for Long-Horizon Robotic Tasks`, `ReWorld`, and `LifeLong-RFT`, where multiple process-like signals or mixed supervision channels are combined to absorb richer task structure. The other is the `boundary / verifier` route, represented by `VLA-RFT`, `WorldEval`, `PRM-as-a-Judge`, and to a lesser extent `World4RL`, where the key move is to stop forcing one scalar to do everything and instead separate dense guidance from correctness checking.

That distinction matters because these papers are not all solving the same problem. Some are trying to build better reward models. Others are arguing for a better decomposition of the full reward-and-evaluation system. So the right comparison questions here are:

- which modules are learned jointly versus separately,
- where failure trajectories enter the training signal,
- whether success is treated as an outcome boundary or as part of dense reward,
- whether the reward stack evaluates robot trajectories, VLA action chunks, or generated world-model rollouts,
- and whether evaluation is folded into reward learning or kept as a separate layer.

| Method | Title | Venue | Date | Code/Project | High-level labels | Key feature/finding |
| --- | --- | --- | --- | --- | --- | --- |
| [Adapt2Reward](https://arxiv.org/abs/2407.14872) | Adapt2Reward: Adapting Video-Language Models to Generalizable Robotic Rewards via Failure Prompts | ECCV 2024 | 7/2024 | - | `Hybrid · Failure-Aware Reward · Success + Failure Prompts · Shaping + RL` | Video-language reward adaptation is strengthened by clustered failure prompts that inject explicit failure structure into reward learning. |
| [RoboReward](https://arxiv.org/abs/2601.00675) | RoboReward: General-Purpose Vision-Language Reward Models for Robotics | Arxiv | 1/2026 | - | `Hybrid · Failure-Aware Reward · Negatives + Near-Misses · RL + Eval` | Large-scale reward dataset and reward VLMs built from success-heavy robot corpora with counterfactual negatives and partial-progress relabeling. |
| [ReWorld](https://arxiv.org/abs/2601.12428) | ReWorld: Multi-Dimensional Reward Modeling for Embodied World Models | Arxiv | 1/2026 | - | `Hybrid · World-Model Reward · Multi-Dimensional Preference · Alignment` | Hierarchical reward model aligns embodied video world models across physical realism, task completion, embodiment plausibility, and visual quality. |
| [LifeLong-RFT](https://arxiv.org/abs/2602.10503) | Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning | Arxiv | 2/2026 | - | `Hybrid · Chunk-Level Process Reward · VLA RFT · Continual Learning` | Multi-dimensional process reward supervises action consistency, continuous trajectory alignment, and format compliance for VLA chunk-level RFT. |
| [Robometer](https://arxiv.org/abs/2603.02115) | Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons | Arxiv | 3/2026 | [Project](https://robometer.github.io/) | `Hybrid · Hybrid Reward · Mixed Process + Preference + Success · Shaping + Termination + Eval` | Hybrid reward stack combining progress, preference, and success supervision. |
| [Large Reward Models](https://arxiv.org/abs/2603.16065) | Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models | Arxiv | 3/2026 | [Project](https://yanru-wu.github.io/Large-Reward-Models/) | `Hybrid · Hybrid Reward · Multi-source Temporal Videos · Shaping + Termination + Eval` | Multi-source VLM reward generator with temporal contrastive, progress, and completion heads. |
| [Generalizable Dense Reward](https://arxiv.org/abs/2604.00055) | Generalizable Dense Reward for Long-Horizon Robotic Tasks | Arxiv | 4/2026 | - | `Hybrid · Hybrid Reward · VLM Progress + Intrinsic Certainty · Shaping + PPO` | VLM-derived progress signals are combined with policy self-certainty to create a generalizable dense reward stack for long-horizon RL fine-tuning. |
| [PRIMO R1](https://arxiv.org/pdf/2603.15600) | From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation | arXiv | 2026-03 | [Project](https://huggingface.co/papers/2603.15600)| `Hybrid · Reasoning Rewarder · Progress + Failure · CoT Monitoring` | RL induces CoT to transform video MLLMs into active critics; initial–current boundary anchoring; strong progress estimation and zero-shot failure detection performance. |
| [VLA-RFT](https://arxiv.org/abs/2510.00406) | VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators | Arxiv | 10/2025 | [Project](https://vla-rft.github.io/) | `Hybrid · Outcome / Verifier · Outcome / Verifier · Shaping + Eval` | Explicit separation between dense shaping rewards and verified outcome rewards. |
| [WorldEval](https://arxiv.org/abs/2505.19017) | WorldEval: World Model as Real-World Robot Policies Evaluator | Arxiv | 5/2025 | [Project](https://worldeval.github.io/) / [Code](https://github.com/liyaxuanliyaxuan/Worldeval) | `Hybrid · Outcome / Verifier · Outcome / Verifier · Eval` | Evaluation and success verification treated as separate system objects. |
| [PRM-as-a-Judge](https://arxiv.org/abs/2603.21669) | PRM-as-a-Judge: A Dense Evaluation Paradigm for Fine-Grained Robotic Auditing | Arxiv | 3/2026 | [Project](https://prm-as-a-judge.github.io/) | `Hybrid · Outcome / Verifier · PRM Judge · Eval / Auditing` | Process reward models are repurposed as dense evaluators for trajectory auditing, progress diagnosis, and failure analysis. |
| [RoboTracer](https://arxiv.org/abs/2512.13660) | RoboTracer: Mastering Spatial Trace with Reasoning in Vision-Language Models for Robotics | Arxiv | 12/2025 | [Project](https://zhoues.github.io/RoboTracer) | `Hybrid · Hybrid Reward · Mixed Process + Preference + Success · Shaping + Eval` | Metric-sensitive process reward combined with outcome-oriented supervision. |
| [World4RL](https://arxiv.org/abs/2509.19080) | World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation | Arxiv | 9/2025 | - | `Hybrid · Outcome / Verifier · Outcome / Verifier · Policy Refinement + Eval` | Outcome-first baseline for policy refinement before richer process and verifier stacks. |


### Expanded Notes

- `Adapt2Reward`: Early and still important failure-aware VLM reward paper. It matters because it explicitly injects clustered failure prompts into reward adaptation instead of learning only from success-style positive evidence.
- `RoboReward`: Important large-scale benchmark-and-model entry for reward learning. It matters because it directly addresses the `success-heavy data` problem through counterfactual negatives, near-misses, and partial-progress relabeling.
- `ReWorld`: Moves hybrid reward modeling from policy trajectories to embodied world-model rollouts. It matters because the reward model evaluates multiple dimensions of generated futures rather than collapsing world-model quality into visual realism alone.
- `LifeLong-RFT`: Adds a chunk-level process-reward variant for VLA post-training. Its reward is not only about task progress, but also about whether discrete actions, decoded continuous trajectories, and output format remain trainable and valid during continual fine-tuning.
- `Robometer`: Most important hybrid reference in this list. Combines intra-trajectory progress supervision, inter-trajectory preference supervision, and success modeling, showing why broader reward systems scale better than progress-only heads.
- `Large Reward Models`: Specializes Qwen3-VL-8B-Instruct into online reward generators via LoRA over a 24-source corpus spanning Open X-Embodiment, HOI4D, EgoDex, LIBERO, and RoboCasa. It explicitly decomposes reward into temporal contrastive, absolute progress, and task completion heads, then uses those signals for zero-shot PPO refinement on ManiSkill3.
- `Generalizable Dense Reward`: Represents the `progress + intrinsic shaping` variant of hybrid reward design. It is important because it combines task-structured progress supervision from a strong vision-language model with policy-side certainty signals instead of relying on preference or verifier data alone.
- `PRIMO R1`: Extends hybrid reward modeling with RL-induced process reasoning. It is important because it jointly models progress estimation and failure detection, rather than separating them into disconnected stages. The initial–current state anchoring design improves temporal state alignment for long-horizon manipulation and helps reduce semantic drift.
- `VLA-RFT`: Makes the `dense shaping + verifier` split explicit inside a simulator-centered training loop. Important because it argues that process reward and verified outcome reward should often remain distinct modules.
- `WorldEval`: Treats evaluation, ranking consistency, and success verification as separate system objects. Useful boundary paper because it clarifies that not every useful reward-like signal should be collapsed into dense process reward.
- `PRM-as-a-Judge`: Extends the verifier branch toward dense auditing rather than policy optimization. Important because it shows that process reward models can also serve as structured judges for progress, efficiency, and failure fingerprints.
- `RoboTracer`: Uses metric-sensitive process reward together with outcome-oriented supervision. Important because it makes `process reward + outcome reward` an additive design choice rather than a false binary.
- `World4RL`: Represents the practical outcome-first path. It is included not because it solves the full process-reward problem, but because it shows the baseline many real systems start from: success-style supervision before richer progress and verifier stacks are introduced.

This section is the most practically important one if your real question is not just "how do I regress reward?" but "how do I build a reward system that survives failed, partial, and out-of-distribution rollouts?" The decisive issue here is often failure ingestion: how failed trajectories enter training, how correctness is verified, and how dense process reward is kept distinct from final outcome judgment.

Read this section as the main correction to overly simple reward-model narratives. In small toy settings, `progress vs delta` can look like the whole problem. In realistic settings, reward modeling usually becomes a layered architecture problem.

Practically, this category becomes necessary as soon as expert-only trajectories stop being enough. If the deployment regime includes partial success, bad recovery behavior, ambiguous end states, or reward hacking risk, then some hybridization or boundary decomposition is usually required. The price is system complexity: more heads, more labels, and more calibration problems between dense and sparse signals.


<a id="reward-models-as-system-interfaces-for-rl-termination-rewind-and-planning"></a>
# **⚙️ Reward Models as System Interfaces for RL, Termination, Rewind, Verification, and Planning**

This category is about `how reward signals are used`, not only about `what reward heads mean`. These papers are included because they make the interface between learned reward and downstream control explicit. Two methods can learn similar reward semantics and still behave very differently if one is used as dense RL shaping while another is used as a stopping rule, a rewind trigger, or a planner-side rollout ranker.

The inclusion rule here is that the paper's main contribution is the `control interface` built around a reward or progress signal. Some of these papers inherit reward semantics from earlier work; their value lies in making the deployment-time use of the signal concrete.

This section also has a useful subdivision. `TT-VLA`, `EVOLVE-VLA`, and `ProgressVLA` are about turning reward-like signals into online adaptation or generation guidance, usually through shaping, differencing, or classifier-style steering. `RoVer`, `CoVer-VLA`, and `SV-VLA` are about test-time verification, action selection, and chunk-level replanning. `EVA` and `World Action Verifier` extend the same interface question to imagined futures and world-model rollouts, where the key issue is whether generated futures are executable and action-reachable rather than merely visually plausible. `SPR` treats progress as a recovery and anomaly-control variable. `AdaPower` treats it as a planner-side ranking signal. These are all "reward uses," but they impose very different requirements on calibration, smoothness, and trustworthiness.

That is why this category is analytically important. A progress model that is good enough for rollout ranking may still be too unstable for termination; a signal that works as dense shaping may still be too noisy for recovery triggers. So comparing interface papers means comparing not just where the score comes from, but what operational burden the downstream loop puts on it.

| Method | Title | Venue | Date | Code/Project | High-level labels | Key feature/finding |
| --- | --- | --- | --- | --- | --- | --- |
| [TT-VLA](https://arxiv.org/abs/2601.06748) | On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning | Arxiv | 1/2026 | - | `Interface · Progress Signal · Inherited Signal + Online Feedback · Shaping + Potential` | Progress estimate converted into reward differences for deployment-time RL adaptation. |
| [EVOLVE-VLA](https://arxiv.org/abs/2512.14666) | EVOLVE-VLA: Test-Time Training from Environment Feedback for Vision-Language-Action Models | Arxiv | 12/2025 | [Project](https://showlab.github.io/EVOLVE-VLA) | `Interface · Progress Signal · Inherited Signal + Online Feedback · Shaping + Termination` | Progress estimator used for both test-time learning and stopping logic. |
| [RoVer](https://arxiv.org/abs/2510.10975) | RoVer: Robot Reward Model as Test-Time Verifier for Vision-Language-Action Model | Arxiv | 10/2025 | - | `Interface · Reward Model as Verifier · PRM + Direction Guidance · Test-Time Verification` | A robot process reward model scores and refines candidate actions at inference time to improve VLA decision-making. |
| [CoVer-VLA](https://arxiv.org/abs/2602.12281) | Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment | Arxiv | 2/2026 | - | `Interface · Verifier Signal · Contrastive Alignment · Test-Time Verification` | A contrastive verifier selects instruction rephrasings and action chunks at test time, making verification itself a scaling axis. |
| [ProgressVLA](https://arxiv.org/abs/2603.27670) | ProgressVLA: Progress-Guided Diffusion Policy for Vision-Language Robotic Manipulation | Arxiv | 3/2026 | - | `Interface · Progress Signal · World Model + Guidance · Sampling + Termination` | A learned progress estimator guides diffusion action sampling through predicted futures and also supports threshold-style stopping. |
| [EVA](https://arxiv.org/abs/2603.17808) | EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards | Arxiv | 3/2026 | [Project](https://eva-project-page.github.io/) | `Interface · World-Model Reward · Inverse Dynamics · Executability` | Inverse dynamics model is repurposed as a reward model to align generated video rollouts with executable robot actions. |
| [World Action Verifier](https://arxiv.org/abs/2604.01985) | World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry | Arxiv | 4/2026 | [Project](https://world-action-verifier.github.io/) / [Code](https://github.com/world-action-verifier/wav_minigrid) | `Interface · World-Model Verifier · State Plausibility + Action Reachability · Active Learning` | Verifies imagined futures through video subgoals, sparse inverse dynamics, and forward-inverse consistency for world-model improvement. |
| [SV-VLA](https://arxiv.org/abs/2604.02965) | Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA | Arxiv | 4/2026 | [Code](https://github.com/edsad122/SV-VLA) | `Interface · Runtime Verifier · Action Chunk Monitoring · Replanning` | Lightweight verifier monitors heavy-VLA action chunks online and triggers replanning when closed-loop reference actions diverge. |
| [SPR](https://arxiv.org/abs/2603.09292) | See, Plan, Rewind: Progress-Aware Vision-Language-Action Models for Robust Robotic Manipulation | Arxiv | 3/2026 | - | `Interface · Progress Signal · Inherited Signal · Recovery / Rewind` | Progress anomalies used as rewind and recovery triggers. |
| [AdaPower](https://arxiv.org/abs/2512.03538) | AdaPower: Specializing World Foundation Models for Predictive Manipulation | Arxiv | 12/2025 | - | `Interface · Progress Signal · Inherited Signal · Planning` | Progress-like reward used to rank imagined rollouts inside an MPC-style loop. |

### Expanded Notes

- `TT-VLA`: Makes the `progress -> reward difference -> RL adaptation` route explicit. Useful because it shows how a learned progress estimator can be converted into deployment-time RL feedback instead of only offline analysis.
- `EVOLVE-VLA`: Uses a learned progress estimator both for test-time learning and for stopping logic, linking reward shaping, accumulative progress estimation, and horizon control inside one system.
- `RoVer`: One of the clearest examples of `reward model as verifier` rather than `reward model as training signal`. It matters because it shows that a process reward model can score and directionally refine action candidates at test time.
- `CoVer-VLA`: Pushes the verifier branch further away from scalar reward regression and toward contrastive alignment retrieval. It matters because it frames verification itself as a scalable deployment-time interface.
- `ProgressVLA`: Uses a progress estimator as a generation-time guidance signal rather than only a scalar monitor. Important because it shows how progress prediction can enter diffusion action synthesis directly through future-state scoring.
- `EVA`: Extends reward-as-verifier to generated video futures. The key point is that an inverse dynamics model can judge whether a visually plausible rollout induces smooth, bounded, embodiment-valid actions.
- `World Action Verifier`: Makes world-model verification explicit by separating plausible future states from action-reachable future states. It is useful because the verifier signal feeds active data acquisition and world-model self-improvement.
- `SV-VLA`: Shows verifier signals as runtime control logic rather than training rewards. It is included because the verifier decides whether an open-loop action chunk can continue executing or must be replaced by a new heavy-model plan.
- `SPR`: Uses progress anomalies to trigger rewind and recovery. Important because it shows that progress can act as a control variable for fault handling, not only as a training reward.
- `AdaPower`: Uses progress-like reward to rank imagined rollouts in an MPC-style loop. Included because it connects reward learning directly to planning-time rollout selection rather than only policy optimization.

This section is where many comparisons become clearer. A paper that looks "similar" at the reward-head level may end up serving a very different role once it enters a closed-loop system. That is why this repository treats downstream use as part of the taxonomy rather than as a secondary engineering detail.

If you are deciding what reward model to build for a real robot stack, this section is often as important as the semantic sections above. In practice, the right reward head depends heavily on whether you need shaping, thresholding, recovery, or planning support.

Practically, this is the most engineering-facing section in the repository. It is where reward modeling stops being just an annotation problem and becomes a systems problem. Many deployment failures are not caused by learning the "wrong" reward semantics, but by using a reasonable signal in the wrong place in the control loop.


<a id="supporting-papers-for-stage-aware-and-temporal-feedback"></a>
# **🧱 Supporting Papers for Stage-Aware and Temporal Feedback**

This final category contains papers that are not the cleanest canonical instances of reward semantics, but they are important for understanding why simple scalar reward formulations often break down on long-horizon tasks. Their shared role is to motivate `stage structure`, `temporally localized supervision`, and richer intermediate feedback.

These papers belong here because they strengthen the motivation for process reward modeling without always fitting neatly into one canonical head type. They help explain why the field moved away from purely endpoint-based supervision and why many modern reward systems need denser intermediate structure.

The distinction here is conceptual rather than architectural. `VLS` is useful because it foregrounds stage geometry and shows why long-horizon manipulation often cannot be reduced to one flat progress scalar. `Keyframe-Guided Structured Rewards` is useful because it shows how stage milestones can be extracted and turned into process reward without hand-written shaping logic. `AIM` is useful because it turns value into a spatial interface between imagined future dynamics and action generation rather than treating value only as a scalar score. `RFTF` is useful because it foregrounds temporally localized feedback and the broader argument for intermediate supervision. None of these papers is the cleanest canonical reward model in this README, but all of them sharpen the motivation for why the canonical categories above exist.

| Method | Title | Venue | Date | Code/Project | High-level labels | Key feature/finding |
| --- | --- | --- | --- | --- | --- | --- |
| [VLS](https://arxiv.org/abs/2602.03973) | VLS: Steering Pretrained Robot Policies via Vision-Language Models | Arxiv | 2/2026 | [Project](https://vision-language-steering.github.io/webpage/) | `Support · Stage / Temporal · Stage-Structured · Steering` | Multi-stage VLM-synthesized programmatic rewards and policy steering for long-horizon tasks. |
| [Keyframe-Guided Structured Rewards](https://arxiv.org/abs/2603.00719) | Keyframe-Guided Structured Rewards for Reinforcement Learning in Long-Horizon Laboratory Robotics | Arxiv | 3/2026 | - | `Support · Stage / Temporal · Structured Keyframes · Shaping` | Demonstration-derived keyframe milestones are converted into structured stage rewards for long-horizon RL. |
| [AIM](https://arxiv.org/abs/2604.11135) | AIM: Intent-Aware Unified world action Modeling with Spatial Value Maps | Arxiv | 4/2026 | - | `Support · Spatial Value · Value-Map Interface · Dense RL Signal` | Spatial value maps bridge future visual dynamics and action generation, with projected value-map responses used as dense rewards during self-distillation RL. |
| [RFTF](https://arxiv.org/abs/2505.19767) | RFTF: Reinforcement Fine-tuning for Embodied Agents with Temporal Feedback | Arxiv | 5/2025 | - | `Support · Stage / Temporal · Temporal Feedback · Shaping` | Temporal and process-oriented intermediate supervision for embodied fine-tuning. |

### Expanded Notes

- `VLS`: Highlights multi-stage VLM-synthesized programmatic rewards and policy steering, reinforcing the claim that long-horizon reward design often benefits from explicit stage structure rather than one flat scalar.
- `Keyframe-Guided Structured Rewards`: Shows a concrete route from demonstrations to structured stage reward. It is useful because it turns milestone extraction and temporal order into a reward-generation mechanism rather than leaving them as informal intuitions.
- `AIM`: Supports the move from scalar progress to spatially grounded value interfaces. It is not a pure reward model, but it shows how value maps can turn future visual predictions into dense control feedback for action learning.
- `RFTF`: Provides direct support for temporal and process-oriented intermediate supervision. Included because it helps justify why dense trajectory-level feedback is worth modeling in the first place.

This section should be read as support material for the rest of the taxonomy. It explains why stage-aware reward geometry and temporally localized feedback keep reappearing whenever tasks become long, compositional, or failure-prone.

Practically, this section is where to look when the main reward-model papers start feeling too implementation-specific. These support papers provide the conceptual pressure that pushes the field toward denser, staged, or temporally localized reward design.


<a id="repository-scope"></a>
## 📌 Repository Scope

This list is intentionally analytical rather than exhaustive.

- It focuses on robotic reward modeling, not all of RL for Embodied AI.
- It prioritizes papers that clarify reward semantics, supervision, failure ingestion, or downstream usage.
- It treats `progress vs delta` as the entry point, not as the full story.

## Star History

<a href="https://www.star-history.com/?repos=VanceChu%2FAwesome-Progress-Reward-Model&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=VanceChu/Awesome-Progress-Reward-Model&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=VanceChu/Awesome-Progress-Reward-Model&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=VanceChu/Awesome-Progress-Reward-Model&type=date&legend=top-left" />
 </picture>
</a>
