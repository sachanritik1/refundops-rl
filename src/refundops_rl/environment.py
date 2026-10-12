"""Headless LLM RL environment with a Gym-like reset/step contract.

No real money or external APIs. One instance represents ONE isolated episode.
This is NOT a Gymnasium subclass (the action/observation spaces are structured
JSON tool calls). A trainer adapter can wrap it for NeMo Gym or another runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .scenarios import RETURN_WINDOW_DAYS, TODAY, Scenario, generate

MAX_STEPS = 8
TOOL_NAMES = ("get_order", "get_payment", "get_policy", "issue_refund", "finish")

# A client can expose these to any OpenAI-compatible tool-calling LLM.
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_order", "description": "Inspect the customer's order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_payment", "description": "Inspect payment/refund ledger for an order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_policy", "description": "Read refund policy", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Issue the entire currently eligible remaining refund in paise; supply an idempotency key", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount_paise": {"type": "integer", "minimum": 1}, "idempotency_key": {"type": "string", "minLength": 1}}, "required": ["order_id", "amount_paise", "idempotency_key"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "finish", "description": "Resolve customer request", "parameters": {"type": "object", "properties": {"resolution": {"type": "string", "enum": ["refunded", "rejected", "already_refunded"]}}, "required": ["resolution"], "additionalProperties": False}}},
]


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class RefundEnvironment:
    def __init__(self, max_steps: int = MAX_STEPS):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self._scenario: Scenario | None = None
        self._refund_ledger_paise = 0
        self._refunds_by_key: dict[str, tuple[int, str]] = {}
        self._observed: set[str] = set()
        self._steps = 0
        self._done = False
        self._trace: list[dict[str, Any]] = []
        self._current_action: Any = None

    def reset(self, *, seed: int, family: str | None = None) -> dict[str, Any]:
        self._scenario = generate(seed, family)
        self._refund_ledger_paise = self._scenario.refunded_paise
        self._refunds_by_key = {}
        self._observed = set()
        self._steps = 0
        self._done = False
        self._trace = []
        self._current_action = None
        return {
            "task": f"A customer asks for the maximum policy-eligible refund for order {self._scenario.order_id}. Check the order, payment, and policy; either issue the exact remaining refund or close the request correctly.",
            "today": TODAY.isoformat(),
            "tools": deepcopy(TOOL_SCHEMAS),
            "remaining_steps": self.max_steps,
        }

    @property
    def trace(self) -> list[dict[str, Any]]:
        return deepcopy(self._trace)

    def _require_reset(self) -> Scenario:
        if self._scenario is None:
            raise RuntimeError("call reset before step")
        return self._scenario

    def _oracle(self) -> tuple[str, int]:
        """Privileged grading rule: NEVER include this in agent observations."""
        s = self._require_reset()
        if s.order_status != "delivered" or s.payment_status != "captured":
            return "rejected", 0
        assert s.delivery_date is not None
        if (TODAY - date.fromisoformat(s.delivery_date)).days > RETURN_WINDOW_DAYS:
            return "rejected", 0
        remaining = s.refundable_paise - s.refunded_paise
        if remaining <= 0:
            return "already_refunded", 0
        return "refunded", remaining

    def _observe(self, tool: str, result: dict[str, Any]) -> dict[str, Any]:
        return {"tool": tool, "result": result, "remaining_steps": self.max_steps - self._steps}

    def _finish_step(self, tool: str, result: dict[str, Any], reward: float = 0.0,
                     terminated: bool = False, *, diagnostic: str | None = None) -> StepResult:
        self._done = terminated or self._steps >= self.max_steps
        truncated = self._steps >= self.max_steps and not terminated
        obs = self._observe(tool, result)
        info: dict[str, Any] = {}
        if self._done:
            info["outcome"] = diagnostic or ("step_limit" if truncated else "completed")
        self._trace.append({"action": deepcopy(self._current_action), "observation": deepcopy(obs), "reward": reward,
                            "terminated": terminated, "truncated": truncated})
        return StepResult(obs, reward, terminated, truncated, info)

    def step(self, action: dict[str, Any]) -> StepResult:
        s = self._require_reset()
        if self._done:
            raise RuntimeError("episode is complete; call reset")
        self._steps += 1
        self._current_action = deepcopy(action)
        tool = action.get("tool") if isinstance(action, dict) else None
        args = action.get("args", {}) if isinstance(action, dict) else None
        if not isinstance(tool, str) or tool not in TOOL_NAMES or not isinstance(args, dict):
            return self._finish_step(str(tool), {"error": "invalid_action"})

        # Arguments may not reference another user's order.
        if tool in ("get_order", "get_payment", "issue_refund") and args.get("order_id") != s.order_id:
            return self._finish_step(tool, {"error": "order_not_found"})

        if tool == "get_order":
            self._observed.add(tool)
            return self._finish_step(tool, {"order_id": s.order_id, "status": s.order_status,
                "amount_paid_paise": s.amount_paid_paise, "shipping_paise": s.shipping_paise,
                "delivered_on": s.delivery_date})

        if tool == "get_payment":
            self._observed.add(tool)
            return self._finish_step(tool, {"status": s.payment_status, "already_refunded_paise": self._refund_ledger_paise})

        if tool == "get_policy":
            self._observed.add(tool)
            return self._finish_step(tool, {"eligible_status": "delivered", "required_payment_status": "captured",
                "return_window_days_inclusive": RETURN_WINDOW_DAYS,
                "nonrefundable": ["shipping"], "refund_rule": "refund all eligible remaining merchandise value"})

        # Payment side effects require evidence from all tools first.
        if tool == "issue_refund":
            if not {"get_order", "get_payment", "get_policy"} <= self._observed:
                return self._finish_step(tool, {"error": "must_inspect_order_payment_and_policy"}, -1.0, True,
                                         diagnostic="unsafe_refund_attempt")
            amount = args.get("amount_paise")
            key = args.get("idempotency_key")
            if type(amount) is not int or amount <= 0 or not isinstance(key, str) or not (1 <= len(key) <= 128):
                return self._finish_step(tool, {"error": "invalid_refund_arguments"}, -1.0, True,
                                         diagnostic="unsafe_refund_attempt")
            if key in self._refunds_by_key:
                old_amount, receipt = self._refunds_by_key[key]
                if old_amount != amount:
                    return self._finish_step(tool, {"error": "idempotency_key_conflict"}, -1.0, True,
                                             diagnostic="unsafe_refund_attempt")
                return self._finish_step(tool, {"status": "already_applied", "receipt": receipt,
                                                "amount_paise": old_amount})
            decision, required_amount = self._oracle()
            if decision != "refunded" or amount != required_amount or self._refund_ledger_paise != s.refunded_paise:
                return self._finish_step(tool, {"error": "ineligible_or_incorrect_amount"}, -1.0, True,
                                         diagnostic="unsafe_refund_attempt")
            # Atomic within this env instance; HTTP adapter uses a per-episode lock.
            receipt = f"RF-{s.order_id}-{len(self._refunds_by_key) + 1}"
            self._refund_ledger_paise += amount
            self._refunds_by_key[key] = (amount, receipt)
            return self._finish_step(tool, {"status": "refunded", "receipt": receipt,
                                            "amount_paise": amount})

        # finish: reward is computed from recorded business state, not a model's claim.
        assert tool == "finish"
        decision = args.get("resolution")
        expected, required = self._oracle()
        evidence_ok = {"get_order", "get_payment", "get_policy"} <= self._observed
        outcome_ok = (decision == expected and evidence_ok and
            (self._refund_ledger_paise == s.refunded_paise + required if expected == "refunded"
             else self._refund_ledger_paise == s.refunded_paise))
        return self._finish_step(tool, {"accepted": bool(outcome_ok)},
            1.0 if outcome_ok else 0.0, True,
            diagnostic="success" if outcome_ok else "incorrect_resolution")
