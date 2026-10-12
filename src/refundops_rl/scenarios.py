"""Seeded, reproducible scenarios. Scenario names are NOT shown to agents.

This is deliberately separate from the agent observation and grader code.
Real training datasets should be larger and split at the scenario-family level.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import random

TODAY = date(2026, 10, 12)
RETURN_WINDOW_DAYS = 30

# Each family has multiple monetary variations by seed; eval has held-out seeds.
FAMILIES = (
    "eligible",
    "partially_refunded",
    "outside_window",
    "uncaptured_payment",
    "undelivered_order",
    "fully_refunded",
)


@dataclass(frozen=True)
class Scenario:
    order_id: str
    amount_paid_paise: int
    shipping_paise: int
    refunded_paise: int
    payment_status: str
    order_status: str
    delivery_date: str | None

    @property
    def refundable_paise(self) -> int:
        return self.amount_paid_paise - self.shipping_paise


def scenario_names() -> tuple[str, ...]:
    return FAMILIES


def generate(seed: int, family: str | None = None) -> Scenario:
    rng = random.Random(seed)
    if family is None:
        family = rng.choice(FAMILIES)
    if family not in FAMILIES:
        raise ValueError(f"unknown scenario family: {family}")

    amount = rng.randrange(12, 61) * 10000  # ₹1,200 .. ₹6,000
    shipping = rng.choice((5000, 7500, 10000, 12500))
    refundable = amount - shipping
    refunded = 0
    payment_status = "captured"
    order_status = "delivered"
    days_ago = rng.randint(1, RETURN_WINDOW_DAYS)

    if family == "partially_refunded":
        refunded = refundable // 2
    elif family == "fully_refunded":
        refunded = refundable
    elif family == "outside_window":
        days_ago = rng.randint(RETURN_WINDOW_DAYS + 1, 90)
    elif family == "uncaptured_payment":
        payment_status = "pending"
    elif family == "undelivered_order":
        order_status = "shipped"

    delivery_date = (TODAY - timedelta(days=days_ago)).isoformat() if order_status == "delivered" else None
    return Scenario(
        order_id=f"ORD-{seed:08d}",
        amount_paid_paise=amount,
        shipping_paise=shipping,
        refunded_paise=refunded,
        payment_status=payment_status,
        order_status=order_status,
        delivery_date=delivery_date,
    )
