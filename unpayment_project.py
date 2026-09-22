from data_loader import load_and_prepare_models
from loan_calculation import calculate_loan_schedule
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Approval & Pricing Portal", page_icon="🏦", layout="wide"
)

with st.spinner("Initializing models and banking pipelines..."):
  models, scaler, feature_columns, num_columns, occupation_mapping = (
      load_and_prepare_models()
  )

# Persistent Session State
if "selected_model" not in st.session_state:
  st.session_state["selected_model"] = "Optimized Random Forest"
if "approval_status" not in st.session_state:
  st.session_state["approval_status"] = None
if "risk_score" not in st.session_state:
  st.session_state["risk_score"] = 0.0

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "1. Model Selection & Benchmark",
        "2. Applicant Eligibility Assessment",
        "3. Loan Pricing & Profitability",
    ],
)

# =========================================================
# SCREEN 1: MODEL SELECTION & BENCHMARK
# =========================================================
if page == "1. Model Selection & Benchmark":
  st.title("Step 1: Credit Risk Model Selection")
  st.write(
      "Compare the two benchmark models and select which one governs the credit"
      " decision engine."
  )

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Logistic Regression")
    m = models["Logistic Regression"]
    st.metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
    st.metric("Recall (Default Catch Rate)", f"{m['recall'] * 100:.2f}%")
    st.metric("F2-Score", f"{m['f2_score'] * 100:.2f}%")
    st.caption(f"Decision Threshold: {m['threshold']:.2f}")

  with col2:
    st.subheader("Optimized Random Forest")
    m = models["Optimized Random Forest"]
    st.metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
    st.metric("Recall (Default Catch Rate)", f"{m['recall'] * 100:.2f}%")
    st.metric("F2-Score", f"{m['f2_score'] * 100:.2f}%")
    st.caption(f"Decision Threshold: {m['threshold']:.2f}")

  st.divider()
  chosen = st.selectbox(
      "Active Model for Production:",
      ["Optimized Random Forest", "Logistic Regression"],
      index=(
          0
          if st.session_state["selected_model"] == "Optimized Random Forest"
          else 1
      ),
  )
  if st.button("Confirm Model Selection", type="primary"):
    st.session_state["selected_model"] = chosen
    st.success(f"Active model updated to: {chosen}")

# =========================================================
# SCREEN 2: APPLICANT ELIGIBILITY ASSESSMENT
# =========================================================
elif page == "2. Applicant Eligibility Assessment":
  st.title("Step 2: Applicant Data & Risk Evaluation")
  st.info(f"Active Decision Model: **{st.session_state['selected_model']}**")

  col1, col2, col3 = st.columns(3)

  with col1:
    st.subheader("Personal")
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 85, 36)
    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow",
        ],
    )
    family_members = st.number_input("Family Members", 1, 10, 2)

  with col2:
    st.subheader("Financial & Job")
    annual_income = st.number_input(
        "Annual Income ($)", 10000.0, 1000000.0, 180000.0, 5000.0
    )
    is_employed = st.radio("Employed?", ["Yes", "No"])
    if is_employed == "Yes":
      years_employed = st.number_input("Years of Employment", 0, 50, 4)
      occupation = st.selectbox(
          "Occupation",
          [
              "Managers",
              "High skill tech staff",
              "Accountants",
              "Core staff",
              "IT staff",
              "HR staff",
              "Laborers",
              "Drivers",
              "Security staff",
              "Cleaning staff",
              "Cooking staff",
              "Low-skill Laborers",
              "Sales staff",
              "Private service staff",
              "Secretaries",
              "Realty agents",
              "Waiters/barmen staff",
              "Unknown",
          ],
      )
    else:
      years_employed = 0
      occupation = "Unemployed"

    type_income = st.selectbox(
        "Income Type",
        ["Working", "Commercial associate", "Pensioner", "State servant"],
    )
    education = st.selectbox(
        "Education Level",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
        ],
    )

  with col3:
    st.subheader("Assets & Contacts")
    car_owner = st.checkbox("Owns Car")
    propert_owner = st.checkbox("Owns Property")
    housing_type = st.selectbox(
        "Housing Type",
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ],
    )
    work_phone = st.checkbox("Work Phone Provided")
    phone = st.checkbox("Home Phone Provided")
    email_id = st.checkbox("Email Provided")

  if st.button("Evaluate Eligibility", type="primary", use_container_width=True):
    type_occupation = occupation_mapping.get(occupation, "Unknown")
    housing_group = (
        housing_type
        if housing_type
        in ["House / apartment", "With parents", "Municipal apartment"]
        else "Other"
    )

    data = {
        "GENDER": 1 if gender == "Female" else 0,
        "Car_Owner": 1 if car_owner else 0,
        "Propert_Owner": 1 if propert_owner else 0,
        "Annual_income": annual_income,
        "Work_Phone": 1 if work_phone else 0,
        "Phone": 1 if phone else 0,
        "EMAIL_ID": 1 if email_id else 0,
        "Family_Members": float(family_members),
        "Age": int(age),
        "Years_Employed": int(years_employed),
        "Is_Unemployed": 1 if is_employed == "No" else 0,
        "Type_Income": type_income,
        "EDUCATION": education,
        "Marital_status": marital_status,
        "Housing_type": housing_group,
        "Type_Occupation": type_occupation,
    }

    input_df = pd.DataFrame([data])
    cat_columns = [
        "Type_Income",
        "EDUCATION",
        "Marital_status",
        "Housing_type",
        "Type_Occupation",
    ]
    input_encoded = pd.get_dummies(
        input_df, columns=cat_columns, drop_first=True, dtype=int
    )
    final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
    final_input[num_columns] = scaler.transform(final_input[num_columns])

    active_cfg = models[st.session_state["selected_model"]]
    prob = active_cfg["model"].predict_proba(final_input)[0, 1]

    st.session_state["risk_score"] = float(prob)
    threshold = active_cfg["threshold"]

    st.divider()
    if prob >= threshold:
      st.session_state["approval_status"] = "DENIED"
      st.error(
          f"❌ Loan Denied: Estimated Default Risk ({prob * 100:.1f}%) exceeds"
          f" the cutoff threshold ({threshold * 100:.0f}%)."
      )
    else:
      st.session_state["approval_status"] = "APPROVED"
      st.success(
          f"✅ Loan Approved: Estimated Default Risk ({prob * 100:.1f}%) meets"
          f" the bank criteria ({threshold * 100:.0f}%)."
      )
      st.info(
          "👉 Proceed to Step 3 (Loan Pricing & Profitability) to configure the"
          " terms."
      )

# =========================================================
# SCREEN 3: LOAN PRICING & PROFITABILITY
# =========================================================
elif page == "3. Loan Pricing & Profitability":
  st.title("Step 3: Loan Terms, Rates & Profit Simulation")

  if st.session_state["approval_status"] is None:
    st.warning("Please evaluate an applicant in Step 2 first.")
  else:
    status_color = (
        "green" if st.session_state["approval_status"] == "APPROVED" else "red"
    )
    st.markdown(
        f"Applicant Status: :{status_color}[**{st.session_state['approval_status']}**]"
        f" | Baseline Default Risk:"
        f" **{st.session_state['risk_score'] * 100:.1f}%**"
    )

    col1, col2 = st.columns(2)

    with col1:
      st.subheader("Loan Structure")
      loan_type = st.selectbox(
          "Product Category",
          [
              "Personal Loan (Unsecured)",
              "Auto Loan (Secured)",
              "Mortgage / Home Equity",
              "Business Expansion",
          ],
      )

      # Suggest typical baseline rates based on loan product
      default_rates = {
          "Personal Loan (Unsecured)": 9.5,
          "Auto Loan (Secured)": 6.0,
          "Mortgage / Home Equity": 4.2,
          "Business Expansion": 8.0,
      }

      amount = st.number_input(
          "Requested Principal ($)", 1000.0, 500000.0, 25000.0, 1000.0
      )
      duration_months = st.slider("Tenure (Months)", 6, 120, 36, 6)
      interest_rate = st.number_input(
          "Annual Interest Rate (%)",
          1.0,
          35.0,
          default_rates[loan_type],
          0.25,
      )

    metrics = calculate_loan_schedule(
        amount,
        interest_rate,
        duration_months,
        st.session_state["risk_score"],
    )

    with col2:
      st.subheader("Borrower Payment Schedule")
      st.metric("Monthly Installment", f"${metrics['monthly_payment']:,.2f}")
      st.metric("Total Repayment", f"${metrics['total_paid']:,.2f}")
      st.metric("Total Interest Generated", f"${metrics['total_interest']:,.2f}")

    st.divider()
    st.subheader("Bank Risk-Adjusted Margin Analysis")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
      st.metric(
          "Expected Loss (Risk * Principal)", f"${metrics['expected_loss']:,.2f}"
      )
    with p_col2:
      profit = metrics["expected_profit"]
      st.metric(
          "Net Expected Bank Profit",
          f"${profit:,.2f}",
          delta="Viable Deal" if profit > 0 else "Unprofitable Deal",
          delta_color="normal" if profit > 0 else "inverse",
      )