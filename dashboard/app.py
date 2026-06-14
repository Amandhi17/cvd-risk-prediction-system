"""
Streamlit Dashboard — Cardiovascular Disease Risk Prediction.

Run locally:
    streamlit run dashboard/app.py

Make sure the FastAPI backend is running first:
    uvicorn api.main:app --reload --port 8000
"""

import os
from pathlib import Path

import google.generativeai as genai
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned.csv"


def get_prevention_tips(patient: dict, risk: str, probability: float,
                        api_key: str) -> str:
    """Call Gemini to generate personalised prevention recommendations."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    chol_map = {1: "normal", 2: "above normal", 3: "well above normal"}
    gluc_map = {1: "normal", 2: "above normal", 3: "well above normal"}
    bmi = patient["weight"] / ((patient["height"] / 100) ** 2)

    prompt = f"""You are a cardiology educator (not a doctor). Based on this
patient profile, give 4 short, specific, actionable lifestyle-and-medical
recommendations to lower their cardiovascular disease risk. Be concrete (e.g.
"target systolic BP below 120 via reduced sodium intake"), not generic
("eat healthier").

Patient profile:
- Age: {patient['age_years']} years
- Gender: {"male" if patient['gender'] == 2 else "female"}
- Height/Weight: {patient['height']} cm / {patient['weight']} kg (BMI {bmi:.1f})
- Systolic / Diastolic BP: {patient['ap_hi']} / {patient['ap_lo']} mmHg
- Cholesterol: {chol_map[patient['cholesterol']]}
- Glucose: {gluc_map[patient['gluc']]}
- Smoker: {"yes" if patient['smoke'] else "no"}
- Drinks alcohol: {"yes" if patient['alco'] else "no"}
- Physically active: {"yes" if patient['active'] else "no"}

Predicted CVD risk: {risk} ({probability:.0%})

Output ONLY a markdown bulleted list of 4 recommendations. End with a one-line
italic disclaimer that this is educational, not medical advice."""

    response = model.generate_content(prompt)
    return response.text


# ---- Page config ---------------------------------------------------------
st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="❤",
    layout="wide",
)
st.title("Cardiovascular Disease Risk Predictor")
st.caption("MLOps pipeline · FastAPI + Gradient Boosting · Kaggle cardio dataset")


# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Backend status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.ok:
            st.success(f"API online — model: {r.json()['model_type']}")
        else:
            st.error("API responded with an error")
    except requests.exceptions.RequestException:
        st.error("API is not reachable. Start it with:\n\n"
                 "`uvicorn api.main:app --reload`")
    st.markdown("---")
    st.markdown("**Endpoints**")
    st.code("GET  /health\nPOST /predict")

    st.markdown("---")
    st.subheader("Prevention Tips (Gemini)")
    gemini_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    if gemini_key:
        st.success("Gemini key loaded — tips will appear after prediction")


# ---- Tabs ----------------------------------------------------------------
tab_predict, tab_insights = st.tabs(["Predict a Patient", "Dataset Insights"])


# =========================================================================
# Tab 1 — Prediction form
# =========================================================================
with tab_predict:
    st.subheader("Enter patient details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Body**")
        age_years = st.slider("Age (years)", 18, 100, 55)
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        height = st.slider("Height (cm)", 120, 220, 168)
        weight = st.slider("Weight (kg)", 30, 200, 70)

    with col2:
        st.markdown("**Blood Pressure**")
        ap_hi = st.slider("Systolic BP (ap_hi)", 70, 250, 120)
        ap_lo = st.slider("Diastolic BP (ap_lo)", 40, 200, 80)
        st.caption(f"Pulse pressure: {ap_hi - ap_lo} mmHg")
        st.markdown("**Lab Results**")
        cholesterol = st.selectbox(
            "Cholesterol",
            options=[1, 2, 3],
            format_func=lambda x: ["Normal", "Above normal",
                                   "Well above normal"][x - 1],
        )
        gluc = st.selectbox(
            "Glucose",
            options=[1, 2, 3],
            format_func=lambda x: ["Normal", "Above normal",
                                   "Well above normal"][x - 1],
        )

    with col3:
        st.markdown("**Lifestyle**")
        smoke = st.radio("Smoker?", ["No", "Yes"], horizontal=True)
        alco = st.radio("Drinks alcohol?", ["No", "Yes"], horizontal=True)
        active = st.radio("Physically active?", ["Yes", "No"], horizontal=True)

    if st.button("Predict CVD Risk", type="primary", use_container_width=True):
        payload = {
            "age_years": float(age_years),
            "gender": 2 if gender == "Male" else 1,
            "height": int(height),
            "weight": float(weight),
            "ap_hi": int(ap_hi),
            "ap_lo": int(ap_lo),
            "cholesterol": int(cholesterol),
            "gluc": int(gluc),
            "smoke": 1 if smoke == "Yes" else 0,
            "alco": 1 if alco == "Yes" else 0,
            "active": 1 if active == "Yes" else 0,
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if r.ok:
                res = r.json()
                risk = res["cvd_risk"]
                prob = res["probability"]
                color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk]
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Level", f"{color} {risk}")
                m2.metric("Probability of CVD", f"{prob:.1%}")
                m3.metric("Model", res["model_used"])
                st.progress(prob)
                if risk == "High":
                    st.error("This patient is at high cardiovascular risk. "
                             "Recommend medical follow-up.")
                elif risk == "Medium":
                    st.warning("Moderate risk — recommend lifestyle review "
                               "and periodic monitoring.")
                else:
                    st.success("Low risk — maintain current healthy habits.")

                if gemini_key and risk in ("High", "Medium"):
                    st.markdown("---")
                    st.subheader("AI-generated prevention tips")
                    with st.spinner("Generating personalised tips..."):
                        try:
                            tips = get_prevention_tips(
                                payload, risk, prob, gemini_key
                            )
                            st.info(tips)
                        except Exception as e:
                            st.warning(f"Could not generate tips: {e}")
                elif not gemini_key and risk in ("High", "Medium"):
                    st.markdown("---")
                    st.caption("Add your Gemini API key in the sidebar to "
                               "get AI-generated prevention tips.")
            else:
                st.error(f"API error: {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")


# =========================================================================
# Tab 2 — Dataset insights
# =========================================================================
with tab_insights:
    st.subheader("Dataset overview")
    if not DATA_PATH.exists():
        st.warning(f"Cleaned data not found at {DATA_PATH}. "
                   "Run `python src/data_cleaning.py` first.")
    else:
        df = pd.read_csv(DATA_PATH)
        df["age_years"] = (df["age"] / 365.25).round(1)
        df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total patients", f"{len(df):,}")
        c2.metric("CVD positive", f"{int(df['cardio'].sum()):,}")
        c3.metric("CVD rate", f"{df['cardio'].mean():.1%}")

        st.markdown("### CVD rate by age group")
        df["age_bucket"] = pd.cut(
            df["age_years"],
            bins=[30, 40, 50, 55, 60, 65, 70],
            labels=["30-40", "40-50", "50-55", "55-60", "60-65", "65-70"],
        ).astype(str)
        st.bar_chart(df.groupby("age_bucket", observed=True)["cardio"].mean())

        st.markdown("### CVD rate by cholesterol level")
        st.bar_chart(df.groupby("cholesterol")["cardio"].mean())

        st.markdown("### CVD rate by glucose level")
        st.bar_chart(df.groupby("gluc")["cardio"].mean())

        st.markdown("### Lifestyle factors")
        life = pd.DataFrame({
            "Smoker": df.groupby("smoke")["cardio"].mean(),
            "Drinker": df.groupby("alco")["cardio"].mean(),
            "Active": df.groupby("active")["cardio"].mean(),
        }).rename(index={0: "No", 1: "Yes"})
        st.dataframe(life.style.format("{:.1%}"))

        with st.expander("Show first 20 rows of cleaned data"):
            st.dataframe(df.head(20))
