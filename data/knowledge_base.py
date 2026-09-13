"""
data/knowledge_base.py
Contains the 12 required Nykaa domain policy documents for Part 1 Task 2.
Each entry contains clean text (2-5 sentences), unique document IDs,
and domain metadata ready for fixed-size and sentence-based chunking.
"""

from typing import List, Dict, Any

KB_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "doc_id": "KB-RET-001",
        "topic": "return_window_by_category",
        "title": "Return Window Policy by Product Category",
        "category": "Returns",
        "text": (
            "Nykaa provides category-specific return windows to balance customer satisfaction with hygiene standards. "
            "Apparel and Footwear items are eligible for return or exchange within 15 calendar days from the date of delivery provided tags are intact. "
            "Beauty, skincare, and cosmetic items must be returned within 5 days and remain sealed in their original protective packaging. "
            "Electronics and personal care appliances cannot be returned once opened, except in documented cases of dead-on-arrival equipment."
        )
    },
    {
        "doc_id": "KB-REF-002",
        "topic": "cod_refund_timelines",
        "title": "Cash on Delivery (COD) Refund Timelines",
        "category": "Refunds",
        "text": (
            "Refunds for Cash on Delivery (COD) orders cannot be remitted back in physical currency. "
            "Customers must submit their verified bank account details (IFSC and Account Number) or UPI VPA via the Nykaa app refund dashboard. "
            "Once the returned parcel clears warehouse quality verification, the COD refund is processed within 3 to 5 business days. "
            "Alternatively, customers may opt for instant Nykaa Wallet credit, which becomes usable immediately upon return approval."
        )
    },
    {
        "doc_id": "KB-SLA-003",
        "topic": "delivery_slas",
        "title": "Standard and Express Delivery SLAs",
        "category": "Shipping",
        "text": (
            "Nykaa orders are dispatched through authorized logistics partners within 24 to 48 hours of order confirmation. "
            "Standard delivery timelines for tier-1 metro cities range between 2 to 4 business days. "
            "Non-metro locations, regional hubs, and North-Eastern states typically require 5 to 7 business days for transit. "
            "In the event of unforeseen regional transit delays exceeding 48 hours past the estimated SLA, customers receive automated SMS tracking alerts."
        )
    },
    {
        "doc_id": "KB-PCK-004",
        "topic": "reverse_pickup_eligibility",
        "title": "Reverse-Pickup Eligibility and Guidelines",
        "category": "Returns",
        "text": (
            "Reverse-pickup services are available across more than 18,000 postal codes serviced by Nykaa logistics affiliates. "
            "A reverse pickup is attempted up to a maximum of three times before the return request is automatically cancelled. "
            "Customers must ensure the item is handed over in its original outer box with all brand tags, freebies, and accessories included. "
            "For pin codes where reverse pickup is unavailable, customers are asked to self-ship, with up to INR 150 reimbursed as store credit."
        )
    },
    {
        "doc_id": "KB-WAR-005",
        "topic": "warranty_terms_by_category",
        "title": "Warranty Coverage Terms by Category",
        "category": "Warranty",
        "text": (
            "All electronic grooming tools, hair straighteners, and facial devices sold on Nykaa carry a standard 1-year to 2-year manufacturer warranty. "
            "Nykaa acts as an authorized retail distributor and does not operate localized technical repair centers directly. "
            "To register or claim warranty coverage, customers must present the original GST invoice generated from their Nykaa order dashboard to the manufacturer service center. "
            "Apparel, Home decor, and Footwear categories are covered exclusively against manufacturing stitch defects reported within 30 days of arrival."
        )
    },
    {
        "doc_id": "KB-CAN-006",
        "topic": "order_cancellation_policy",
        "title": "Order Cancellation Conditions",
        "category": "Orders",
        "text": (
            "Orders can be cancelled directly through the Nykaa app or website before they are processed for shipment at our fulfillment centers. "
            "Once an order transitions to the 'Shipped' status, in-app cancellation is locked, and the customer must reject the package at the doorstep upon arrival. "
            "Prepaid orders cancelled prior to dispatch receive an automatic reversal to the original source payment method within 24 to 48 hours."
        )
    },
    {
        "doc_id": "KB-LOY-007",
        "topic": "loyalty_points_redemption",
        "title": "Nykaa Reward Points and Loyalty Redemption",
        "category": "Loyalty",
        "text": (
            "Nykaa Reward Points are credited to the customer account upon the successful completion and delivery of an eligible order. "
            "Points can be redeemed at checkout against up to 20 percent of the total payable basket value for non-discounted merchandise. "
            "Reward points maintain an expiration validity of 365 calendar days from the date of initial issuance. "
            "In cases where an order is returned or cancelled, any points accrued from that transaction are debited from the reward balance."
        )
    },
    {
        "doc_id": "KB-PAY-008",
        "topic": "payment_failure_retry",
        "title": "Payment Failure, Deductions, and Retry Policy",
        "category": "Payments",
        "text": (
            "If funds are debited from a customer bank account or credit card but the Nykaa order confirmation fails, the transaction is marked pending. "
            "Inter-bank reconciliation engines automatically flag unassigned settlements within 24 hours of checkout interruption. "
            "The debited amount is returned to the original source card, bank, or UPI account within 5 to 7 banking days without manual intervention. "
            "Customers are encouraged to verify network stability and initiate a retry only after checking their 'My Orders' screen."
        )
    },
    {
        "doc_id": "KB-EXC-009",
        "topic": "size_exchange_policy",
        "title": "Size Exchange Guidelines for Apparel and Footwear",
        "category": "Returns",
        "text": (
            "Size exchanges are supported exclusively on eligible Apparel and Footwear items within 7 days of verified delivery. "
            "An exchange request is dependent upon the real-time stock availability of the requested alternate size in our warehouses. "
            "Only one size exchange is permitted per purchased line item; subsequent sizing concerns must be treated as standard return returns. "
            "Items submitted for exchange must be unworn, unwashed, and accompanied by the original manufacturer security ribbon."
        )
    },
    {
        "doc_id": "KB-DAM-010",
        "topic": "damaged_item_claim",
        "title": "Damaged or Tampered Package Claim Process",
        "category": "Claims",
        "text": (
            "Any parcel received in a physically damaged, unsealed, or visibly tampered condition must be reported to Nykaa support within 48 hours of delivery. "
            "Customers are strictly required to upload clear high-resolution images or an unboxing video showing the outer shipping label, box condition, and damaged contents. "
            "Our internal fraud detection and transit team conducts an audit with courier manifests within 72 hours of submission. "
            "Upon validation, Nykaa arranges an immediate priority replacement dispatch or issues a complete refund."
        )
    },
    {
        "doc_id": "KB-INT-011",
        "topic": "international_shipping_restrictions",
        "title": "Cross-Border and International Shipping Terms",
        "category": "Shipping",
        "text": (
            "Nykaa currently offers international dispatch only to selected destination countries and specific overseas partner territories. "
            "Aerosols, flammable perfumes, pressure-packed nail paints, and certain lithium-battery electronics are legally barred from air cargo export. "
            "International consignments are non-returnable, and exchange requests cannot be fulfilled outside the domestic territory of India. "
            "Customs import duties and local border clearing tariffs are not included at checkout and remain the sole responsibility of the recipient."
        )
    },
    {
        "doc_id": "KB-ESC-012",
        "topic": "support_escalation_matrix",
        "title": "Customer Support Escalation Tier Matrix",
        "category": "Support",
        "text": (
            "Level 1 assistance is managed through automated chat assistance and front-line customer service executives handling standard inquiries. "
            "If an issue concerning delayed orders, refunds, or damaged claims remains unresolved past 48 hours, it is escalated to Level 2 Operations Supervisors. "
            "Level 3 Grievance Redressal can be addressed directly to the appointed Nodal Grievance Officer via official email ticketing. "
            "The Nodal Officer operates under statutory guidelines and guarantees formal resolution within 15 business days of ticket receipt."
        )
    }
]

def load_documents_for_indexing() -> List[Dict[str, Any]]:
    """Returns the documents pre-formatted for ChromaDB insertion."""
    documents = []
    for entry in KB_DOCUMENTS:
        documents.append({
            "id": entry["doc_id"],
            "text": entry["text"],
            "metadata": {
                "doc_id": entry["doc_id"],
                "topic": entry["topic"],
                "title": entry["title"],
                "category": entry["category"]
            }
        })
    return documents

if __name__ == "__main__":
    docs = load_documents_for_indexing()
    print(f"Loaded {len(docs)} knowledge-base documents successfully.")
    for d in docs:
        sentences = [s for s in d["text"].split(". ") if s.strip()]
        print(f"[{d['id']}] Topic: {d['metadata']['topic']} | Sentences: {len(sentences)}")