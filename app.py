import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Credit Approval Simulator", page_icon="🏦", layout="wide")

@st.cache_resource
def load_and_train_data():
    df1 = pd.read_csv("Credit_card.csv")
    df2 = pd.read_csv("Credit_card_label.csv")
    df = pd.merge(df1, df2, how="inner", on="Ind_ID")

    df = df.dropna(subset=["GENDER"]).copy()

    df.loc[(df["Type_Occupation"].isnull()) & (df["Employed_days"] > 0), "Type_Occupation"] = "Unemployed"

    occupation_mapping = {
        "Managers": "High_Skill_Office",
        "High skill tech staff": "High_Skill_Office",
        "Accountants": "High_Skill_Office",
        "Core staff": "High_Skill_Office",
        "IT staff": "High_Skill_Office",
        "HR staff": "High_Skill_Office",
        "Laborers": "Blue_Collar",
        "Drivers": "Blue_Collar",
        "Security staff": "Blue_Collar",
        "Cleaning staff": "Blue_Collar",
        "Cooking staff": "Blue_Collar",
        "Low-skill Laborers": "Blue_Collar",
        "Sales staff": "Services_Sales",
        "Private service staff": "Services_Sales",
        "Secretaries": "Services_Sales",
        "Realty agents": "Services_Sales",
        "Waiters/barmen staff": "Services_Sales",
        "Unemployed": "Unemployed"
    }
    df["Type_Occupation"] = df["Type_Occupation"].map(occupation_mapping).fillna("Unknown")

    df["Car_Owner"] = df["Car_Owner"].map({"Y": 1, "N": 0})
    df["Propert_Owner"] = df["Propert_Owner"].map({"Y": 1, "N": 0})
    df["GENDER"] = df["GENDER"].map({"F": 1, "M": 0})

    df["Housing_type"] = df["Housing_type"].apply(
        lambda x: x if x in ["House / apartment", "With parents", "Municipal apartment"] else "Other"
    )

    df["Age"] = -df["Birthday_count"] // 365
    df["Age"] = df["Age"].fillna(df["Age"].mean()).round().astype(int)

    df["Years_Employed"] = np.where(df["Employed_days"] > 0, 0, -df["Employed_days"] // 365)
    df["Is_Unemployed"] = (df["Employed_days"] > 0).astype(int)

    df = df.drop(columns=["Ind_ID", "Employed_days", "Birthday_count", "Mobile_phone", "CHILDREN"])

    q1_income = df["Annual_income"].quantile(0.25)
    q3_income = df["Annual_income"].quantile(0.75)
    iqr_income = q3_income - q1_income
    limit_income = q3_income + 2.5 * iqr_income
    df["Annual_income"] = np.where(df["Annual_income"] > limit_income, 2 * q3_income, df["Annual_income"])
    df["Annual_income"] = df["Annual_income"].fillna(df["Annual_income"].median())

    q1_family = df["Family_Members"].quantile(0.25)
    q3_family = df["Family_Members"].quantile(0.75)
    iqr_family = q3_family - q1_family
    limit_family = q3_family + 2.5 * iqr_family
    df["Family_Members"] = np.where(df["Family_Members"] > limit_family, 2 * q3_family, df["Family_Members"])

    cat_columns = ["Type_Income", "EDUCATION", "Marital_status", "Housing_type", "Type_Occupation"]
    df["EDUCATION"] = df["EDUCATION"].replace({"Academic degree": "Higher education"})
    df = pd.get_dummies(df, columns=cat_columns, drop_first=True, dtype=int)

    X = df.drop("label", axis=1)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_columns = ["Annual_income", "Family_Members", "Age", "Years_Employed"]
    scaler = StandardScaler()
    X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
    X_test_scaled = X_test.copy()
    X_test_scaled[num_columns] = scaler.transform(X_test_scaled[num_columns])

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced_subsample", random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
    }

    test_predictions = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        test_predictions[name] = m.predict_proba(X_test_scaled)[:, 1]

    return models, scaler, list(X_train.columns), num_columns, occupation_mapping, test_predictions

with st.spinner("Training models and preparing data..."):
    models, scaler, feature_columns, num_columns, occupation_mapping, test_predictions = load_and_train_data()

st.title("Credit Decision & Policy Simulator")

st.sidebar.header("Model & Risk Configuration")

chosen_model_name = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Logistic Regression", "Decision Tree"]
)

threshold = st.sidebar.slider(
    "Maximum Acceptable Risk Threshold (%)",
    min_value=1,
    max_value=90,
    value=42,
    step=1
) / 100.0

current_model = models[chosen_model_name]
test_probs = test_predictions[chosen_model_name]
acceptance_rate = (test_probs < threshold).mean() * 100
default_detection_rate = (test_probs >= threshold).mean() * 100

st.sidebar.markdown("---")
st.sidebar.subheader("Portfolio Simulation")
st.sidebar.metric("Overall Acceptance Rate", f"{acceptance_rate:.1f}%")
st.sidebar.metric("Rejected Applications", f"{default_detection_rate:.1f}%")

st.write("Complete the applicant profile to evaluate credit eligibility based on your risk policy.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", min_value=18, max_value=85, value=35)
    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
    )
    family_members = st.number_input("Family Members", min_value=1, max_value=10, value=2)

with col2:
    st.subheader("Financial & Employment")
    annual_income = st.number_input(
        "Annual Income ($)",
        min_value=10000.0,
        max_value=1000000.0,
        value=180000.0,
        step=5000.0
    )
    is_employed = st.radio("Employed?", ["Yes", "No"])
    
    if is_employed == "Yes":
        years_employed = st.number_input("Years of Employment", min_value=0, max_value=50, value=4)
        occupation = st.selectbox(
            "Occupation",
            [
                "Managers", "High skill tech staff", "Accountants", "Core staff",
                "IT staff", "HR staff", "Laborers", "Drivers", "Security staff",
                "Cleaning staff", "Cooking staff", "Low-skill Laborers", "Sales staff",
                "Private service staff", "Secretaries", "Realty agents",
                "Waiters/barmen staff", "Unknown"
            ]
        )
    else:
        years_employed = 0
        occupation = "Unemployed"

    type_income = st.selectbox(
        "Income Type",
        ["Working", "Commercial associate", "Pensioner", "State servant"]
    )
    education = st.selectbox(
        "Education",
        ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"]
    )

with col3:
    st.subheader("Assets & Verification")
    car_owner = st.checkbox("Car Owner")
    propert_owner = st.checkbox("Property Owner")
    housing_type = st.selectbox(
        "Housing Type",
        [
            "House / apartment", "With parents", "Municipal apartment",
            "Rented apartment", "Office apartment", "Co-op apartment"
        ]
    )
    work_phone = st.checkbox("Work Phone Available")
    phone = st.checkbox("Home Phone Available")
    email_id = st.checkbox("Email Available")

if st.button("Evaluate Application", type="primary", use_container_width=True):
    type_occupation = occupation_mapping.get(occupation, "Unknown")
    housing_group = housing_type if housing_type in ["House / apartment", "With parents", "Municipal apartment"] else "Other"

    applicant_data = {
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
        "Type_Occupation": type_occupation
    }

    input_df = pd.DataFrame([applicant_data])

    cat_columns = ["Type_Income", "EDUCATION", "Marital_status", "Housing_type", "Type_Occupation"]
    input_encoded = pd.get_dummies(input_df, columns=cat_columns, drop_first=True, dtype=int)

    final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
    final_input[num_columns] = scaler.transform(final_input[num_columns])

    default_risk = current_model.predict_proba(final_input)[0, 1]

    st.divider()

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Predicted Default Risk", f"{default_risk * 100:.1f}%")
    with m_col2:
        st.metric("Model Cutoff Threshold", f"{threshold * 100:.0f}%")

    if default_risk >= threshold:
        st.error(f"Loan Denied: Risk ({default_risk * 100:.1f}%) exceeds the accepted policy threshold ({threshold * 100:.0f}%).")
    else:
        st.success(f"Loan Approved: Risk ({default_risk * 100:.1f}%) is within the accepted policy threshold ({threshold * 100:.0f}%).")