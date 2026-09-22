import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st


@st.cache_resource
def load_and_prepare_models():
  df1 = pd.read_csv("Credit_card.csv")
  df2 = pd.read_csv("Credit_card_label.csv")
  df = pd.merge(df1, df2, how="inner", on="Ind_ID")

  df = df.dropna(subset=["GENDER"]).copy()
  df.loc[(df["Type_Occupation"].isnull()) & (df["Employed_days"] > 0), "Type_Occupation",] = "Unemployed"

  occupation_mapping = {"Managers": "High_Skill_Office", "High skill tech staff": "High_Skill_Office", "Accountants": "High_Skill_Office", "Core staff": "High_Skill_Office", "IT staff": "High_Skill_Office", "HR staff": "High_Skill_Office",
                        "Laborers": "Blue_Collar", "Drivers": "Blue_Collar", "Security staff": "Blue_Collar", "Cleaning staff": "Blue_Collar", "Cooking staff": "Blue_Collar", "Low-skill Laborers": "Blue_Collar",
                        "Sales staff": "Services_Sales", "Private service staff": "Services_Sales", "Secretaries": "Services_Sales", "Realty agents": "Services_Sales", "Waiters/barmen staff": "Services_Sales",
                        "Unemployed": "Unemployed",
  }
  df["Type_Occupation"] = (df["Type_Occupation"].map(occupation_mapping).fillna("Unknown"))

  df["Car_Owner"] = df["Car_Owner"].map({"Y": 1, "N": 0})
  df["Propert_Owner"] = df["Propert_Owner"].map({"Y": 1, "N": 0})
  df["GENDER"] = df["GENDER"].map({"F": 1, "M": 0})

  df["Housing_type"] = df["Housing_type"].apply(lambda x: (x if x in ["House / apartment", "With parents", "Municipal apartment"] else "Other"))

  df["Age"] = -df["Birthday_count"] // 365
  df["Age"] = df["Age"].fillna(df["Age"].mean()).round().astype(int)

  df["Years_Employed"] = np.where(df["Employed_days"] > 0, 0, -df["Employed_days"] // 365)
  df["Is_Unemployed"] = (df["Employed_days"] > 0).astype(int)

  df = df.drop(columns=["Ind_ID", "Employed_days", "Birthday_count", "Mobile_phone", "CHILDREN"])

  q1_inc = df["Annual_income"].quantile(0.25)
  q3_inc = df["Annual_income"].quantile(0.75)
  iqr_inc = q3_inc - q1_inc
  limit_inc = q3_inc + 2.5 * iqr_inc
  df["Annual_income"] = np.where(df["Annual_income"] > limit_inc, 2 * q3_inc, df["Annual_income"])
  df["Annual_income"] = df["Annual_income"].fillna(df["Annual_income"].median())

  q1_fam = df["Family_Members"].quantile(0.25)
  q3_fam = df["Family_Members"].quantile(0.75)
  iqr_fam = q3_fam - q1_fam
  limit_fam = q3_fam + 2.5 * iqr_fam
  df["Family_Members"] = np.where(df["Family_Members"] > limit_fam, 2 * q3_fam, df["Family_Members"])

  cat_columns = ["Type_Income", "EDUCATION", "Marital_status", "Housing_type", "Type_Occupation"]
  df["EDUCATION"] = df["EDUCATION"].replace({"Academic degree": "Higher education"})

  df = pd.get_dummies(df, columns=cat_columns, drop_first=True, dtype=int)

  X = df.drop("label", axis=1)
  y = df["label"].astype(int)

  X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)

  num_columns = ["Annual_income", "Family_Members", "Age", "Years_Employed"]
  scaler = StandardScaler()
  X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
  X_test_scaled = X_test.copy()
  X_test_scaled[num_columns] = scaler.transform(X_test_scaled[num_columns])

  lr_model = LogisticRegression(max_iter=1000, random_state=42)
  lr_model.fit(X_train, y_train)

  rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42)
  rf_model.fit(X_train, y_train)

  lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
  rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

  lr_preds = (lr_probs >= 0.08).astype(int)
  rf_preds = (rf_probs >= 0.42).astype(int)

  model_pack = {
      "Logistic Regression": {
          "model": lr_model,
          "threshold": 0.08,
          "accuracy": accuracy_score(y_test, lr_preds),
          "recall": recall_score(y_test, lr_preds, zero_division=0),
          "f2_score": fbeta_score(
              y_test, lr_preds, beta=2, zero_division=0
          ),
      },
      "Optimized Random Forest": {
          "model": rf_model,
          "threshold": 0.42,
          "accuracy": accuracy_score(y_test, rf_preds),
          "recall": recall_score(y_test, rf_preds, zero_division=0),
          "f2_score": fbeta_score(
              y_test, rf_preds, beta=2, zero_division=0
          ),
      },
  }

  return (model_pack, scaler, list(X_train.columns), num_columns, occupation_mapping)