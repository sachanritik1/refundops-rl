"""Tiny CPU-only RL experiment: tabular Q-learning for the final decision.

The agent's first three evidence-gathering tool calls are scripted. Only the
final choice (refund / reject / already_refunded) is learned. This shows an
actual update to policy parameters, but is NOT LLM RL or multi-step GRPO.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import random

from .environment import RefundEnvironment
from .scenarios import FAMILIES

CHOICES = ("refunded", "rejected", "already_refunded")
State = tuple[str, str, bool, bool]


def observe(env: RefundEnvironment, seed: int, family: str) -> tuple[State, str, int]:
    initial = env.reset(seed=seed, family=family)
    order_id = initial["task"].split("order ")[1].split(".")[0]
    order = env.step({"tool": "get_order", "args": {"order_id": order_id}}).observation["result"]
    payment = env.step({"tool": "get_payment", "args": {"order_id": order_id}}).observation["result"]
    policy = env.step({"tool": "get_policy", "args": {}}).observation["result"]

    in_window = (order["delivered_on"] is not None and
        (date.fromisoformat(initial["today"]) - date.fromisoformat(order["delivered_on"])).days <= policy["return_window_days_inclusive"])
    remaining = order["amount_paid_paise"] - order["shipping_paise"] - payment["already_refunded_paise"]
    state = (order["status"], payment["status"], in_window, remaining > 0)
    return state, order_id, remaining


def play(env: RefundEnvironment, order_id: str, remaining: int, action: str, seed: int) -> float:
    if action == "refunded":
        refund = env.step({"tool": "issue_refund", "args": {"order_id": order_id,
            "amount_paise": max(1, remaining), "idempotency_key": f"toy-{seed}"}})
        if refund.terminated or refund.truncated:
            return refund.reward
    result = env.step({"tool": "finish", "args": {"resolution": action}})
    return result.reward


def trial(q: dict[State, list[float]], *, seed: int, family: str,
          rng: random.Random, epsilon: float) -> tuple[State, int, float]:
    env = RefundEnvironment()
    state, order_id, remaining = observe(env, seed, family)
    values = q.setdefault(state, [0.0, 0.0, 0.0])
    if rng.random() < epsilon:
        choice = rng.randrange(len(CHOICES))
    else:
        choice = max(range(len(CHOICES)), key=lambda i: (values[i], -i))
    return state, choice, play(env, order_id, remaining, CHOICES[choice], seed)


def evaluate(q: dict[State, list[float]], episodes: int = 300) -> dict[str, float | int]:
    rng = random.Random(9182)
    wins = unsafe = 0
    for i in range(episodes):
        _, _, reward = trial(q, seed=100000 + i, family=FAMILIES[i % len(FAMILIES)],
                             rng=rng, epsilon=0.0)
        wins += reward == 1.0
        unsafe += reward == -1.0
    return {"episodes": episodes, "success_rate": round(wins / episodes, 4),
            "unsafe_action_rate": round(unsafe / episodes, 4)}


def train(episodes: int = 1800, alpha: float = 0.3, epsilon: float = 0.2) -> tuple[dict, dict, dict]:
    rng = random.Random(12345)
    q: dict[State, list[float]] = {}
    before = evaluate(q)
    # Ephemeral state, terminal reward: contextual-bandit special case of RL.
    # Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))
    for i in range(episodes):
        family = FAMILIES[i % len(FAMILIES)]
        state, choice, reward = trial(q, seed=i, family=family, rng=rng, epsilon=epsilon)
        q[state][choice] += alpha * (reward - q[state][choice])
    after = evaluate(q)
    return q, before, after


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny tabular RL policy on RefundOps")
    parser.add_argument("--episodes", type=int, default=1800)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    q, before, after = train(episodes=args.episodes)
    print(json.dumps({"algorithm": "tabular Q-learning / contextual bandit",
        "note": "Learns only final outcome, not the tool sequence or LLM weights",
        "train_episodes": args.episodes, "learned_states": len(q),
        "held_out_before_training": before, "held_out_after_training": after}, indent=2))


if __name__ == "__main__":
    main()
