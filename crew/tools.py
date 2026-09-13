from typing import Dict, Any
from data.dataset import ORDERS

def check_order_status(record_id: str) -> Dict[str, Any]:
    """
    Looks up order details and evaluates a designed escalation score.
    
    Formula:
    escalation_score = 0.6 * (1 if delayed else 0) + 0.4 * (days_since_created / 30.0)
    
    Threshold justification:
    A score > 0.70 signifies either a delayed shipment with an age over 8 days,
    or a non-delayed order waiting beyond 24 days (the 80th percentile of days_since_created).
    """
    normalized_id = record_id.strip().upper()
    order = next((o for o in ORDERS if o["record_id"] == normalized_id), None)
    
    if not order:
        return {"error": f"Order {record_id} not found."}
    
    recency_factor = order["days_since_created"] / 30.0
    delay_penalty = 1.0 if order["delayed_shipment"] else 0.0
    
    escalation_score = round((0.6 * delay_penalty) + (0.4 * recency_factor), 3)
    
    return {
        "record_id": order["record_id"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "days_since_created": order["days_since_created"],
        "delayed_shipment": order["delayed_shipment"],
        "escalation_score": escalation_score,
        "recommend_escalation": escalation_score >= 0.70
    }