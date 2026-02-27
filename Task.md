# Task: Implement G-PPO Algorithm

## Context
Combine PPO and GRPO:
-   Use PPO framework (Actor/Critic).
-   Broadcast external reward (like GRPO).
-   Remove KL penalty from reward (use as auxiliary loss like GRPO).
-   Fit Value Model using Sigmoid loss on external signals (0/1 outcomes).
-   Advantage $A_t = R_{ext} - V(s_{t-1})$ where $R_{ext}$ is the broadcasted external reward (0/1 outcome), for t>1 (skipping bos).
-   Multiple rollouts (like GRPO).

## References
- GRPO: Group Relative Policy Optimization
- Standard PPO with GAE (Generalized Advantage Estimation)

## Key Files
- `verl/trainer/ppo/core_algos.py` - AdvantageEstimator enum (line 88), advantage functions, value loss
- `verl/trainer/ppo/ray_trainer.py` - `compute_advantage()` dispatcher (line 182)
- `verl/workers/critic/dp_critic.py` - Critic update with value loss (line 228)
- `verl/trainer/config/algorithm.py` - AlgoConfig dataclass (line 60)

## Checklist

### Planning
- [x] ~~Create implementation plan `implementation_plan.md`~~ (incorporated into this file) <!-- id: 1 -->

### Implementation
- [x] Add `AdvantageEstimator.GPPO` to `core_algos.py` <!-- id: 2 -->
- [x] Implement `compute_gppo_advantage` in `core_algos.py` <!-- id: 3 --> (depends on: 2)
    -   Broadcasts sparse rewards to all tokens.
    -   Computes $A_t = R_{ext} - V(s_{t-1})$ with shifted values:
        -   Shift values tensor by 1: `values_shifted[t] = values[t-1]`
        -   First response token (t=0) has no baseline, so mask it out.
        -   `advantage[t] = R_ext - values_shifted[t]` for t > 0.
    -   Returns `(advantages, returns)` where `returns = R_ext` (broadcasted binary reward for sigmoid loss target).
- [x] Add Sigmoid Value Loss support to `core_algos.py` <!-- id: 4 -->
    -   Extend `compute_value_loss` with `loss_type` parameter (default: `"mse"`).
    -   When `loss_type="bce"`: Binary cross-entropy loss using `F.binary_cross_entropy_with_logits(vpreds, returns)`
    -   Where `returns` contains the binary outcome signal (0/1).
    -   Keep existing MSE logic as default for backward compatibility.
- [x] Handle KL penalty as auxiliary loss in policy update <!-- id: 5 -->
    -   Configured via `use_kl_loss=true` and `kl_loss_coef` in actor config (existing infrastructure).
- [x] Update `ray_trainer.py` to handle `GPPO` estimator <!-- id: 6 --> (depends on: 2, 3, 4)
    -   Added GPPO case in `compute_advantage` function.
    -   Passes `values` to `compute_gppo_advantage`.
- [x] Verify `dp_critic.py` passes correct arguments for sigmoid loss <!-- id: 7 --> (depends on: 4)
    -   Updated both `dp_critic.py` and `megatron_critic.py` to pass `loss_type`.
- [x] Add GPPO configuration options to training config <!-- id: 8 -->
    -   `adv_estimator: "gppo"` in AlgoConfig
    -   `value_loss_type: str = "mse"` in AlgoConfig (new field, options: `"mse"`, `"bce"`)
    -   Added `value_loss_type` to `critic.yaml` config

### Verification
- [ ] Unit test `compute_gppo_advantage` with mock rewards/values <!-- id: 9 -->
- [ ] Unit test sigmoid value loss computation <!-- id: 10 -->
- [ ] Integration test with small model dry run <!-- id: 11 -->

### Recipe Creation
- [x] Create G-PPO recipe in `recipe/gppo/` <!-- id: 12 -->
    -   Created `recipe/gppo/train_gppo.sh`
    -   Based on `test_gspo.sh` structure for logging/paths
    -   Key config:
        -   `algorithm.adv_estimator=gppo`
        -   `algorithm.value_loss_type=bce`
        -   Critic enabled (like PPO)
        -   Multiple rollouts `actor_rollout_ref.rollout.n=8` (like GRPO)
        -   `algorithm.use_kl_in_reward=false`, `use_kl_loss=true` (KL as auxiliary loss)

# Integrating Math-Verify

## Implementation Plan

### Reward Function
- **[NEW] `verl/utils/reward_score/math_verify_reward.py`**
  - Import `parse`, `verify` from `math_verify`.
  - Implement `compute_score(solution_str, ground_truth)`:
    - Parse `ground_truth` using `gold = parse(ground_truth)`.
    - Parse `solution_str` using `answer = parse(solution_str)`.
    - Function should gracefully handle `ImportError` (return 0.0 or log warning).
    - Return `1.0` if `verify(gold, answer)` is True, else `0.0`.
  - Reference: `https://github.com/huggingface/Math-Verify` usage example.

### Configuration
- **[MODIFY] `ablation/run_gppo.sh`**
  - Update `custom_reward_function.path` to point to the new reward file.
  - Update `custom_reward_function.name` to `compute_score`.

## Checklist
- [ ] Create `verl/utils/reward_score/math_verify_reward.py`
    - [ ] Implement `compute_score` using `math_verify`
    - [ ] Add fallback/error handling
- [ ] Modify `ablation/run_gppo.sh`
    - [ ] Update `custom_reward_function` paths

# Integrating Rollout Probability to GPPO Advantage

## Context
The goal is to make the rollout probability (specifically, the log probabilities of the generated tokens) available within the `compute_gppo_advantage` function. This will allow using these probabilities in the advantage calculation logic.

## Implementation Plan

### `verl`

#### [MODIFY] [verl/verl/trainer/ppo/core_algos.py](file:///Users/zenofficial/Documents/llm/gppo/verl/verl/trainer/ppo/core_algos.py)

- Update `compute_gppo_advantage` signature to accept `token_level_log_probs: Optional[torch.Tensor] = None`.
- Update the docstring.

#### [MODIFY] [verl/verl/trainer/ppo/ray_trainer.py](file:///Users/zenofficial/Documents/llm/gppo/verl/verl/trainer/ppo/ray_trainer.py)

- In `compute_advantage` function, when `adv_estimator == AdvantageEstimator.GPPO`:
  - extract `token_level_log_probs` from `data.batch`.
  - Prefer `data.batch["rollout_log_probs"]` if available (this corresponds to the original log probs from rollout), utilizing `actor_rollout_ref.rollout.calculate_log_probs=True`.
  - Pass this tensor to `core_algos.compute_gppo_advantage`.

## Checklist
- [ ] Implement `core_algos.py` changes <!-- id: 13 -->
- [ ] Implement `ray_trainer.py` changes <!-- id: 14 -->
- [ ] Verify plumbing with a test run <!-- id: 15 -->

# Reward Adjustment via IPW

## Context
The user requested to use Inverse Probability Weighting (IPW) to adjust the external reward $R_{ext}$. This utilizes the `rollout_log_probs` (or `old_log_probs`) passed to `compute_gppo_advantage`.

## Implementation Plan

### `verl`

#### [MODIFY] [verl/verl/trainer/ppo/core_algos.py](file:///Users/zenofficial/Documents/llm/gppo/verl/verl/trainer/ppo/core_algos.py)

- Modify `compute_gppo_advantage` to:
    1.  Calculate probability of rollout for each token: `token_probs = exp(token_level_log_probs)`.
    2.  (Pending) Use these probabilities to adjust `R_ext`.

## Checklist
- [x] implementation_plan.md updated with IPW formula <!-- id: 16 -->
- [x] `compute_gppo_advantage` updated with probability calculation <!-- id: 17 -->