def calculate_loan_schedule(amount, annual_rate_pct, years, model_pred_risk, t_ref_years=2.0):
    months = int(years * 12)
    monthly_rate = (annual_rate_pct / 100.0) / 12.0

    # Amortización mensual (cuota fija francesa)
    if monthly_rate > 0:
        monthly_payment = (amount * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1))
    else:
        monthly_payment = amount / months

    total_paid = monthly_payment * months
    total_interest = total_paid - amount

    clamped_risk = min(max(model_pred_risk, 0.0), 0.999)
    cumulative_risk_prob = 1.0 - (1.0 - clamped_risk) ** (years / t_ref_years)
    cumulative_risk_prob = min(max(cumulative_risk_prob, 0.0), 1.0)

    expected_loss = amount * cumulative_risk_prob
    expected_profit = (total_interest * (1.0 - cumulative_risk_prob)) - expected_loss

    return {"months": months, "monthly_payment": monthly_payment, "total_paid": total_paid, "total_interest": total_interest, "expected_profit": expected_profit,
            "expected_loss": expected_loss, "cumulative_risk_prob": cumulative_risk_prob, "annual_equivalent_risk": 1.0 - (1.0 - clamped_risk) ** (1.0 / t_ref_years)}