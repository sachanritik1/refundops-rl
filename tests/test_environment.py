from datetime import date
import pytest

from refundops_rl.environment import RefundEnvironment
from refundops_rl.scenarios import FAMILIES, TODAY
from refundops_rl.benchmark import run_episode


def setup(family="eligible", seed=42, max_steps=8):
    env = RefundEnvironment(max_steps=max_steps)
    initial = env.reset(seed=seed, family=family)
    order_id = initial["task"].split("order ")[1].split(".")[0]
    return env, initial, order_id


def inspect(env, order_id):
    order = env.step({"tool": "get_order", "args": {"order_id": order_id}}).observation["result"]
    payment = env.step({"tool": "get_payment", "args": {"order_id": order_id}}).observation["result"]
    policy = env.step({"tool": "get_policy", "args": {}}).observation["result"]
    return order, payment, policy


@pytest.mark.parametrize("family", FAMILIES)
def test_expert_succeeds_each_family(family):
    for seed in (0, 13, 100091):
        outcome = run_episode(seed, family, "scripted")
        assert outcome["success"], (seed, family, outcome)
        assert outcome["reward"] == 1.0


def test_weak_baseline_is_not_perfect():
    scores = [run_episode(i, family, "reject_all")["success"] for i, family in enumerate(FAMILIES)]
    assert 0 < sum(scores) < len(scores)


def test_same_seed_reproduces_task_and_steps():
    a, obs_a, ida = setup(seed=99)
    b, obs_b, idb = setup(seed=99)
    assert obs_a == obs_b
    assert a.step({"tool": "get_order", "args": {"order_id": ida}}) == b.step({"tool": "get_order", "args": {"order_id": idb}})


def test_reset_clears_all_mutation():
    env, _, order_id = setup()
    order, payment, _ = inspect(env, order_id)
    amount = order["amount_paid_paise"] - order["shipping_paise"]
    env.step({"tool": "issue_refund", "args": {"order_id": order_id, "amount_paise": amount, "idempotency_key": "k"}})
    env.reset(seed=42, family="eligible")
    again = env.step({"tool": "get_payment", "args": {"order_id": order_id}})
    assert again.observation["result"]["already_refunded_paise"] == payment["already_refunded_paise"]


def test_wrong_amount_refused_without_side_effects():
    env, _, order_id = setup()
    order, _, _ = inspect(env, order_id)
    amount = order["amount_paid_paise"] - order["shipping_paise"]
    result = env.step({"tool": "issue_refund", "args": {"order_id": order_id, "amount_paise": amount + 1, "idempotency_key": "k"}})
    assert result.reward == -1 and result.terminated
    assert env._refund_ledger_paise == 0


def test_refund_requires_evidence():
    env, _, order_id = setup()
    result = env.step({"tool": "issue_refund", "args": {"order_id": order_id, "amount_paise": 1, "idempotency_key": "k"}})
    assert result.reward == -1 and result.terminated


def test_idempotent_retries_and_conflicts():
    env, _, order_id = setup()
    order, payment, _ = inspect(env, order_id)
    amount = order["amount_paid_paise"] - order["shipping_paise"]
    action = {"tool": "issue_refund", "args": {"order_id": order_id, "amount_paise": amount, "idempotency_key": "retry-1"}}
    first = env.step(action)
    second = env.step(action)
    assert first.observation["result"]["receipt"] == second.observation["result"]["receipt"]
    assert second.observation["result"]["status"] == "already_applied"
    assert env._refund_ledger_paise == amount
    # A changed payload with the same key is always rejected.
    conflict = env.step({"tool": "issue_refund", "args": {**action["args"], "amount_paise": amount + 1}})
    assert conflict.reward == -1 and conflict.terminated
    assert env._refund_ledger_paise == amount


def test_no_forged_finish_reward():
    env, _, _ = setup()
    result = env.step({"tool": "finish", "args": {"resolution": "refunded"}})
    assert result.reward == 0 and result.terminated


def test_expired_refund_unsafe_attempt():
    env, _, order_id = setup("outside_window")
    order, _, _ = inspect(env, order_id)
    amount = order["amount_paid_paise"] - order["shipping_paise"]
    outcome = env.step({"tool": "issue_refund", "args": {"order_id": order_id, "amount_paise": amount, "idempotency_key": "k"}})
    assert outcome.reward == -1 and outcome.terminated


def test_episode_step_limit():
    env, _, order_id = setup(max_steps=2)
    a = env.step({"tool": "get_order", "args": {"order_id": order_id}})
    b = env.step({"tool": "get_order", "args": {"order_id": order_id}})
    assert not a.terminated and not a.truncated
    assert not b.terminated and b.truncated
    with pytest.raises(RuntimeError):
        env.step({"tool": "get_policy", "args": {}})


def test_no_cross_order_access():
    env, _, _ = setup()
    result = env.step({"tool": "get_order", "args": {"order_id": "ORD-OTHER"}})
    assert result.observation["result"] == {"error": "order_not_found"}


def test_finishing_does_not_expose_oracle_details():
    env, _, order_id = setup()
    inspect(env, order_id)
    out = env.step({"tool": "finish", "args": {"resolution": "rejected"}})
    assert out.reward == 0.0
    assert "expected" not in str(out.observation)


def test_toy_policy_learns_from_rewards():
    from refundops_rl.train_toy_policy import train
    q, before, after = train(episodes=240)
    assert after["success_rate"] > before["success_rate"]
    assert after["success_rate"] >= 0.90
    assert len(q) > 0
