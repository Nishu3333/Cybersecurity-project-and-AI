def score_row(row: dict) -> tuple[float, int, list]:
    reasons = []
    s = 0.10  # base

    amount = float(row.get("Amount", 0) or 0)
    if amount >= 1_000_000:
        s += 0.30; reasons.append("High amount ≥ NPR 1,000,000")
    elif amount >= 500_000:
        s += 0.15; reasons.append("Large amount ≥ NPR 500,000")
    elif amount >= 100_000:
        s += 0.05

    if str(row.get("RiskRating","")).lower() == "high":
        s += 0.15; reasons.append("Customer risk rating: High")
    elif str(row.get("RiskRating","")).lower() == "medium":
        s += 0.05

    if int(row.get("LateNightTxn", 0)) == 1:
        s += 0.10; reasons.append("Late night transaction")

    if int(row.get("IsHighRiskCountry", 0)) == 1:
        s += 0.20; reasons.append("High-risk counterparty country")

    if str(row.get("Channel","")).lower() == "online" and amount >= 300_000:
        s += 0.05; reasons.append("Large online transfer")

    s = max(0.0, min(0.95, s))
    pred = 1 if s >= 0.70 else 0
    if pred == 1 and not reasons:
        reasons.append("Multiple weak indicators combined")
    return s, pred, reasons
