# RefundOps RL

A **small, production-minded RL environment** for teaching tool-using LLM agents to handle refund operations. Runs locally, with **no paid model, GPU, payment gateway, or real customer data**.

It implements the **environment and rollout/evaluation layer**, plus a **tiny optional tabular RL training demonstration**. It does **not** fine-tune an LLM. A genuine LLM training pipeline would pass scored rollouts to an RL trainer such as GRPO, which updates language-model weights.

## Why this is an RL environment (not just a dataset)

An agent gets a goal, chooses tools, observes results, and changes state over multiple steps. Every episode begins from a repeatable seeded scenario. A private verifier grades **actual money-ledger state**, rather than trusting what the agent claims. It can be reset and run repeatedly across many different scenarios. The same core can collect scored agent rollouts for training or evaluate a fixed model.

```
LLM agent -> tool-call -> RefundEnvironment.step(action)
              ^                |
              |             tool output
              +----------------+
                        |
                  private verifier -> reward
                        |
               rollout JSONL -> future GRPO trainer
```

## Requirements

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

### Run the benchmark

```bash
python -m refundops_rl.benchmark --episodes 60 --policy scripted --split eval --output examples/rollouts.jsonl
python -m refundops_rl.benchmark --episodes 60 --policy reject_all --split eval --output examples/weak-rollouts.jsonl
```

The scripted baseline fetches order, payment and policy, calculates the refund, and finishes appropriately. `reject_all` deliberately makes a naive incorrect assumption. The benchmark writes individual tool-call trajectories as JSONL.

**Important:** These baselines do **not** train a neural model; they give us known-good and known-bad trajectories to verify the environment.

### Actually train a tiny policy (CPU, no GPU)

```bash
python -m refundops_rl.train_toy_policy --episodes 1800
```

This optional exercise performs **real tabular Q-learning parameter updates** using environment rewards and prints success rates on held-out seeds *before vs. after*. It learns only the **final refund/reject decision** after three scripted information-gathering calls. It is a contextual-bandit-like learning demo, **not** a trained LLM and **not** GRPO. The production-style environment is the reusable part; genuine LLM tool-use RL is the next milestone.

### Run as a local HTTP service

```bash
uvicorn refundops_rl.server:app --reload --host 127.0.0.1 --port 8000
```

Create an episode:

```bash
curl -s http://127.0.0.1:8000/v1/episodes \
  -H 'content-type: application/json' \
  -d '{"seed":42,"family":"eligible"}'
```

Copy `episode_id` and the order ID in the task into the next requests:

```bash
curl -s http://127.0.0.1:8000/v1/episodes/EPISODE_ID/step \
  -H 'content-type: application/json' \
  -d '{"tool":"get_order","args":{"order_id":"ORD-00000042"}}'
```

Other calls:

```json
{"tool":"get_payment","args":{"order_id":"ORD-00000042"}}
{"tool":"get_policy","args":{}}
{"tool":"issue_refund","args":{"order_id":"ORD-00000042","amount_paise":150000,"idempotency_key":"episode-42"}}
{"tool":"finish","args":{"resolution":"refunded"}}
```

The refund amount above is **illustrative**; calculate the correct amount from order/payment/policy data. The server refuses wrong amounts. With `--reload`, sessions may be lost across reloads.

Swagger UI: http://127.0.0.1:8000/docs

Or using Docker:

```bash
docker build -t refundops-rl .
docker run --rm -p 127.0.0.1:8000:8000 refundops-rl
```

## Agent contract

```python
from refundops_rl import RefundEnvironment

env = RefundEnvironment(max_steps=8)
obs = env.reset(seed=42, family="eligible")
# Send obs["task"] and obs["tools"] to your tool-calling LLM.
# Then call env.step({"tool": tool_name, "args": parsed_tool_args})
# Return the resulting observation as a tool result, repeat until terminated or truncated.
```

`step()` returns `StepResult(observation, reward, terminated, truncated, info)`.

| Tool | What it does |
|---|---|
| `get_order` | Read order state, paid amount, delivery and shipping |
| `get_payment` | Read payment capture and previously refunded amounts |
| `get_policy` | Read eligibility rules and refund window |
| `issue_refund` | Atomically simulate a refund, checking policy and idempotency |
| `finish` | Declare refunded / rejected / already_refunded |

**Reward:** `+1` only for a correct final resolution and matching internal refund ledger; `0` for unfinished or incorrect benign outcomes; `-1` for unsafe refund attempts. A step limit produces `truncated=True` and no success reward.

The verifier checks the underlying ledger *and* that the agent actually inspected the order/payment/policy. It does not leak privileged expected outcomes through the tool observations.

## Scenario generation and generalization

Six families: eligible, partially refunded, expired return window, uncaptured payment, undelivered order, and fully refunded. Amounts, shipping and delivery age vary deterministically by seed. `--split eval` uses a disjoint seed range from `train`; for serious experiments, add **held-out scenario families and policies**, not just held-out random seeds.

## What “production-minded” means here

Included: typed Python interfaces, strict API request models, reproducible episodes, isolated per-session state, a step limit, no real side effects, integer money amounts (paise), policy/ledger-based rewards, idempotency keys, per-session locks, session expiry and cap, tests, CI, Docker, health route and JSONL trace output.

**Not production deployment-ready:** HTTP endpoint has no authentication or TLS; state is process-local and is lost on restart; the task generator is small; there is no observability stack, persistence, distributed locking, queue, real tool-calling model, adversarial grader audit, or RL trainer. Do not expose this server publicly or use it with live payments. For multiple app workers or replicas, use durable storage plus database transactions/compare-and-swap. See `docs/roadmap.md`.

## Suggested learning path

1. Run both benchmarks, inspect a trajectory. Change a policy rule and add tests.
2. Connect a real tool-calling model to this `reset/step` loop. Compare models on the same held-out cases.
3. Generate larger train/eval datasets; add faults (timeouts, ambiguous state, partial failures, retries).
4. Build an adapter to a maintained environment/training framework (e.g. NeMo Gym); train a small open-source model with GRPO and plot held-out pass rate *before vs after*.
