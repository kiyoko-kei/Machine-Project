import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

#Load saved artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model  = joblib.load(os.path.join(BASE_DIR, "best_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
le     = joblib.load(os.path.join(BASE_DIR, "label_encoder.joblib"))

FEATURE_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education.num",
    "marital.status", "occupation", "relationship", "race", "sex",
    "capital.gain", "capital.loss", "hours.per.week", "native.country"
]

#Label Mappings
WORKCLASS_MAP = {
    0: "Federal-gov",
    1: "Local-gov",
    2: "Private",
    3: "Self-emp-inc",
    4: "Self-emp-not-inc",
    5: "State-gov",
    6: "Without-pay"
}

EDUCATION_MAP = {
    0:  "10th",
    1:  "11th",
    2:  "12th",
    3:  "1st-4th",
    4:  "5th-6th",
    5:  "7th-8th",
    6:  "9th",
    7:  "Assoc-acdm",
    8:  "Assoc-voc",
    9:  "Bachelors",
    10: "Doctorate",
    11: "HS-grad",
    12: "Masters",
    13: "Preschool",
    14: "Prof-school",
    15: "Some-college"
}

MARITAL_MAP = {
    0: "Divorced",
    1: "Married-AF-spouse",
    2: "Married-civ-spouse",
    3: "Married-spouse-absent",
    4: "Never-married",
    5: "Separated",
    6: "Widowed"
}

OCCUPATION_MAP = {
    0:  "Adm-clerical",
    1:  "Armed-Forces",
    2:  "Craft-repair",
    3:  "Exec-managerial",
    4:  "Farming-fishing",
    5:  "Handlers-cleaners",
    6:  "Machine-op-inspct",
    7:  "Other-service",
    8:  "Priv-house-serv",
    9:  "Prof-specialty",
    10: "Protective-serv",
    11: "Sales",
    12: "Tech-support",
    13: "Transport-moving"
}

RELATIONSHIP_MAP = {
    0: "Husband",
    1: "Not-in-family",
    2: "Other-relative",
    3: "Own-child",
    4: "Unmarried",
    5: "Wife"
}

RACE_MAP = {
    0: "Amer-Indian-Eskimo",
    1: "Asian-Pac-Islander",
    2: "Black",
    3: "Other",
    4: "White"
}

NATIVE_COUNTRY_MAP = {
    0:  "Cambodia",       1:  "Canada",          2:  "China",
    3:  "Columbia",       4:  "Cuba",             5:  "Dominican-Republic",
    6:  "Ecuador",        7:  "El-Salvador",      8:  "England",
    9:  "France",         10: "Germany",          11: "Greece",
    12: "Guatemala",      13: "Haiti",            14: "Holand-Netherlands",
    15: "Honduras",       16: "Hong",             17: "Hungary",
    18: "India",          19: "Iran",             20: "Ireland",
    21: "Italy",          22: "Jamaica",          23: "Japan",
    24: "Laos",           25: "Mexico",           26: "Nicaragua",
    27: "Outlying-US(Guam-USVI-etc)",             28: "Peru",
    29: "Philippines",    30: "Poland",           31: "Portugal",
    32: "Puerto-Rico",    33: "Scotland",         34: "South",
    35: "Taiwan",         36: "Thailand",         37: "Trinadad&Tobago",
    38: "United-States",  39: "Vietnam",          40: "Yugoslavia"
}

#Prescriptive Logic
def get_recommendation(pred):
    if pred == 0:
        return {
            "Predicted Income": "<= $50K",
            "Priority": "HIGH",
            "Action": (
                "Refer to government financial aid programs; enroll in upskilling "
                "or vocational training; provide financial literacy workshops."
            ),
            "Justification": (
                "Predicted to earn at or below $50K annually. "
                "Targeted support and education can improve their economic trajectory."
            )
        }
    else:
        return {
            "Predicted Income": "> $50K",
            "Priority": "LOW",
            "Action": (
                "Offer investment planning consultation; recommend wealth management "
                "and tax optimization services."
            ),
            "Justification": (
                "Predicted to earn above $50K annually. "
                "Proactive financial planning can sustain and grow their wealth."
            )
        }

#UI
st.set_page_config(page_title="Income Predictor", layout="centered")
st.title("Income Classification & Decision Support Tool")
st.markdown(
    "Predict whether an individual earns **<= $50K or > $50K** per year "
    "and receive actionable recommendations."
)
st.divider()

with st.form("input_form"):
    st.subheader("Enter Individual Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=35)

        workclass = st.selectbox(
            "Work Class",
            options=list(WORKCLASS_MAP.keys()),
            format_func=lambda x: WORKCLASS_MAP[x]
        )

        fnlwgt = st.number_input(
            "Final Weight (fnlwgt)",
            min_value=10000, max_value=1500000, value=189778,
            help="Sampling weight — how many people in the population this record represents."
        )

        education = st.selectbox(
            "Education Level",
            options=list(EDUCATION_MAP.keys()),
            format_func=lambda x: EDUCATION_MAP[x],
            index=11
        )

        education_num = st.slider(
            "Education Years  (1 = Preschool  →  16 = Doctorate)",
            min_value=1, max_value=16, value=10
        )

        marital_status = st.selectbox(
            "Marital Status",
            options=list(MARITAL_MAP.keys()),
            format_func=lambda x: MARITAL_MAP[x]
        )

        hours_per_week = st.slider("Hours Worked per Week", min_value=1, max_value=99, value=40)

    with col2:
        occupation = st.selectbox(
            "Occupation",
            options=list(OCCUPATION_MAP.keys()),
            format_func=lambda x: OCCUPATION_MAP[x]
        )

        relationship = st.selectbox(
            "Relationship",
            options=list(RELATIONSHIP_MAP.keys()),
            format_func=lambda x: RELATIONSHIP_MAP[x]
        )

        race = st.selectbox(
            "Race",
            options=list(RACE_MAP.keys()),
            format_func=lambda x: RACE_MAP[x]
        )

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        capital_gain = st.number_input(
            "Capital Gain ($)",
            min_value=0, max_value=99999, value=0,
            help="Income from investments, aside from wages/salary."
        )

        capital_loss = st.number_input(
            "Capital Loss ($)",
            min_value=0, max_value=4356, value=0,
            help="Losses from investments."
        )

        native_country = st.selectbox(
            "Native Country",
            options=list(NATIVE_COUNTRY_MAP.keys()),
            format_func=lambda x: NATIVE_COUNTRY_MAP[x],
            index=38
        )

    submitted = st.form_submit_button("Predict & Get Recommendation")

#Output
if submitted:
    input_data = np.array([[
        age, workclass, fnlwgt, education, education_num,
        marital_status, occupation, relationship, race, sex,
        capital_gain, capital_loss, hours_per_week, native_country
    ]])
    input_df     = pd.DataFrame(input_data, columns=FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    rec = get_recommendation(prediction)

    st.divider()
    st.subheader("Prediction Result")

    with st.expander("View Selected Input Summary"):
        summary = {
            "Age":             age,
            "Work Class":      WORKCLASS_MAP[workclass],
            "Education":       EDUCATION_MAP[education],
            "Education Years": education_num,
            "Marital Status":  MARITAL_MAP[marital_status],
            "Occupation":      OCCUPATION_MAP[occupation],
            "Relationship":    RELATIONSHIP_MAP[relationship],
            "Race":            RACE_MAP[race],
            "Sex":             "Female" if sex == 0 else "Male",
            "Capital Gain":    f"${capital_gain:,}",
            "Capital Loss":    f"${capital_loss:,}",
            "Hours/Week":      hours_per_week,
            "Native Country":  NATIVE_COUNTRY_MAP[native_country],
            "Final Weight":    fnlwgt
        }
        for k, v in summary.items():
            st.markdown(f"**{k}:** {v}")

    if prediction == 0:
        st.error(f"**Predicted Income Class:** {rec['Predicted Income']}")
    else:
        st.success(f"**Predicted Income Class:** {rec['Predicted Income']}")

    st.subheader("📋 Prescriptive Recommendation")
    st.markdown(f"**Priority Level:** {rec['Priority']}")
    st.info(f"**Recommended Action:** {rec['Action']}")
    st.caption(rec["Justification"])