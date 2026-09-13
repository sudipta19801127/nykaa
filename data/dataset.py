"""
data/dataset.py
Deterministic synthetic order dataset generator for Nykaa support agent.
Satisfies Part 1 Task 1.
"""

import random
from typing import List, Dict, Any, Optional

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]


def generate_orders(seed: int = 42, count: int = 45) -> List[Dict[str, Any]]:
    """
    Generates deterministic orders satisfying:
    - Count per category >= 3
    - Count per status >= 1
    - delayed_shipment between 10% and 30%
    """
    random.seed(seed)
    orders = []

    # Category pricing range rationale: Nykaa product catalogue ranges from
    # modest cosmetic items (INR 250) to high-end gadgets and skincare appliances (INR 8,500).
    price_ranges = {
        "Beauty": (250, 2500),
        "Apparel": (500, 4500),
        "Footwear": (800, 6000),
        "Home": (400, 5000),
        "Electronics": (1200, 8500),
    }

    # Ensure minimum guarantees first
    for cat in CATEGORIES:
        for _ in range(3):
            orders.append({"category": cat})
    for st in STATUSES:
        orders.append({"status": st})

    # Fill remaining to reach target count
    while len(orders) < count:
        orders.append({})

    final_orders = []
    delayed_count = 0

    for idx, stub in enumerate(orders):
        category = stub.get("category", random.choice(CATEGORIES))
        status = stub.get(
            "status",
            random.choices(STATUSES, weights=[0.25, 0.30, 0.25, 0.10, 0.10])[0],
        )
        p_min, p_max = price_ranges[category]
        order_val = random.randint(p_min // 50, p_max // 50) * 50
        days = random.randint(0, 30)

        # Tune probability of delayed shipment to hit 10% - 30% band
        is_delayed = random.random() < 0.20
        if is_delayed:
            delayed_count += 1

        record = {
            "record_id": f"NYK-{1000 + idx}",
            "category": category,
            "product_category": category,  # Alias for compatibility across tools
            "status": status,
            "order_value_inr": order_val,
            "days_since_created": days,
            "delayed_shipment": is_delayed,
        }
        final_orders.append(record)

    # Validation assertions
    cat_counts = {c: sum(1 for o in final_orders if o["category"] == c) for c in CATEGORIES}
    stat_counts = {s: sum(1 for o in final_orders if o["status"] == s) for s in STATUSES}
    delay_ratio = delayed_count / len(final_orders)

    assert all(cnt >= 3 for cnt in cat_counts.values()), "Category minimum violated"
    assert all(cnt >= 1 for cnt in stat_counts.values()), "Status coverage violated"
    assert 0.10 <= delay_ratio <= 0.30, f"Delay ratio {delay_ratio:.2f} out of bounds"

    return final_orders


# Generate primary dataset
ORDERS = generate_orders(seed=42, count=45)
ORDERS_DATASET = ORDERS

# Map for O(1) order lookup
ORDERS_BY_ID: Dict[str, Dict[str, Any]] = {
    o["record_id"].upper(): o for o in ORDERS
}


def get_order_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an order record by ID (e.g. 'NYK-1002'), case-insensitive."""
    return ORDERS_BY_ID.get(record_id.strip().upper())


if __name__ == "__main__":
    print(f"Generated {len(ORDERS)} orders.")
    print(f"Delay Ratio: {sum(1 for o in ORDERS if o['delayed_shipment']) / len(ORDERS):.1%}")
    print("Sample record (NYK-1004):", get_order_by_id("NYK-1004"))