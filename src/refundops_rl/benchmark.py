"""Collect verifiable rollouts; compare a scripted expert with a weak baseline.

Neither baseline updates model weights. JSONL output is a dataset *input* for
future training integration, not itself a trained model.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from .environment import RefundEnvironment
from .scenarios import FAMILIES


def run_episode(seed: int, family: str, policy: str) -> dict[str, Any]:
    env = RefundEnvironment()
    initial = env.reset(seed=seed, family=family)
    order_id = initial["task"].split("order ")[1].split(".")[0]
    steps: list[dict[str, Any]] = []
    facts: dict[str, dict[str, Any]] = {}

    def take(tool: str, args: dict[str, Any]) -> None:
        result = env.step({"tool": tool, "args": args})
        steps.append({"action": {"tool": tool, "args": args},
            "observation": result.observation, "reward": result.reward,
            "terminated": result.terminated, "truncated": result.truncated})
        facts[tool] = result.observation.get("result", {})

    for tool, args in (
        ("get_order", {"order_id": order_id}),
        ("get_payment", {"order_id": order_id}),
        ("get_policy", {}),
    ):
        take(tool, args)

    if policy == "reject_all":
        take("finish", {"resolution": "rejected"})
    elif policy == "scripted":
        o = facts["get_order"]
        p = facts["get_payment"]
        pol = facts["get_policy"]
        eligible = (
            o["status"] == pol["eligible_status"]
            and p["status"] == pol["required_payment_status"]
            and o["delivered_on"] is not None
            and (date.fromisoformat(initial["today"]) - date.fromisoformat(o["delivered_on"])).days <= pol["return_window_days_inclusive"]
        )
        left = o["amount_paid_paise"] - o["shipping_paise"] - p["already_refunded_paise"]
        if not eligible:
            take("finish", {"resolution": "rejected"})
        elif left <= 0:
            take("finish", {"resolution": "already_refunded"})
        else:
            take("issue_refund", {"order_id": order_id, "amount_paise": left,
                "idempotency_key": f"episode-{seed}-refund"})
            take("finish", {"resolution": "refunded"})
    else:
        raise ValueError("policy must be 'scripted' or 'reject_all'")

    return {"scenario_family": family, "seed": seed, "policy": policy,
            "initial_observation": initial, "steps": steps,
            "reward": sum(s["reward"] for s in steps),
            "success": steps[-1]["reward"] == 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RefundOps RL trajectories")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--policy", choices=["scripted", "reject_all"], default="scripted")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--output", type=Path, default=Path("examples/rollouts.jsonl"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    seed_base = 100000 if args.split == "eval" else 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wins = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for i in range(args.episodes):
            rollout = run_episode(seed=seed_base + i, family=FAMILIES[i % len(FAMILIES)], policy=args.policy)
            wins += rollout["success"]
            fh.write(json.dumps(rollout, separators=(",", ":")) + "\n")
    print(json.dumps({"policy": args.policy, "split": args.split, "episodes": args.episodes,
                      "success_rate": round(wins / args.episodes, 4), "file": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
