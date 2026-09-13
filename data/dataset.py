import random
from typing import List, Dict, Any

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
        status = stub.get("status", random.choices(
            STATUSES, weights=[0.25, 0.30, 0.25, 0.10, 0.10]
        )[0])
        p_min, p_max = price_ranges[category]
        order_val = random.randint(p_min // 50, p_max // 50) * 50
        days = random.randint(0, 30)

        # Tune probability of delayed shipment to hit 10% - 30% band
        is_delayed = random.random() < 0.20
        if is_delayed:
            delayed_count += 1

        final_orders.append({
            "record_id": f"NYK-{1000 + idx}",
            "category": category,
            "status": status,
            "order_value_inr": order_val,
            "days_since_created": days,
            "delayed_shipment": is_delayed,
        })

    # Validation assertions
    cat_counts = {c: sum(1 for o in final_orders if o["category"] == c) for c in CATEGORIES}
    stat_counts = {s: sum(1 for o in final_orders if o["status"] == s) for s in STATUSES}
    delay_ratio = delayed_count / len(final_orders)

    assert all(cnt >= 3 for cnt in cat_counts.values()), "Category minimum violated"
    assert all(cnt >= 1 for cnt in stat_counts.values()), "Status coverage violated"
    assert 0.10 <= delay_ratio <= 0.30, f"Delay ratio {delay_ratio:.2f} out of bounds"

    return final_orders

ORDERS = generate_orders()

if __name__ == "__main__":
    print(f"Generated {len(ORDERS)} orders.")
    print(f"Delay Ratio: {sum(1 for o in ORDERS if o['delayed_shipment']) / len(ORDERS):.1%}")