def calculate_loan_schedule(amount, annual_rate_pct, months, risk_prob):
  monthly_rate = (annual_rate_pct / 100.0) / 12.0
  if monthly_rate > 0:
    monthly_payment = (amount * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1))
  else:
    monthly_payment = amount / months

  total_paid = monthly_payment * months
  total_interest = total_paid - amount

  expected_loss = amount * risk_prob
  expected_profit = (total_interest * (1.0 - risk_prob)) - expected_loss

  return {"monthly_payment": monthly_payment, "total_paid": total_paid,"total_interest": total_interest, "expected_profit": expected_profit, "expected_loss": expected_loss,}