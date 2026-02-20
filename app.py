import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction AI", page_icon="🏥", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    for col in ["Glucose", "BloodPressure", "BMI"]:
        df[col] = df[col].replace(0, None)
        df[col] = df[col].fillna(df[col].mean())
    df = df.drop(columns=["SkinThickness", "Insulin"])
    features = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
    X = df[features]
    Y = df["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    accuracy = accuracy_score(Y_test, model.predict(X_test))
    return model, accuracy, features

model, accuracy, features = train_model()

st.title("🏥 Diabetes Prediction AI")
st.markdown("**By Jad Mrad** · Computer Engineering Student · Lebanon 🇱🇧")
st.markdown(f"Model Accuracy: **{round(accuracy * 100, 2)}%**")
st.markdown("---")

st.subheader("Enter Patient Information")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 70)

with col2:
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 100, 25)

st.markdown("---")

if st.button("🔍 Predict Diabetes", use_container_width=True):
    patient = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, bmi, dpf, age]],
        columns=features
    )
    result = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0]

    if result == 1:
        st.error("⚠️ Result: High risk of Diabetes detected")
        st.metric("Diabetes Probability", f"{round(probability[1] * 100, 1)}%")
    else:
        st.success("✅ Result: Low risk of Diabetes")
        st.metric("Healthy Probability", f"{round(probability[0] * 100, 1)}%")

    # Added disclaimer here so it shows up right after the prediction
    st.warning("⚠️ **Disclaimer:** I am continuously working to improve the accuracy of this model. Currently, the results are not 100% accurate and this tool should not be used as a substitute for professional medical advice.")

    st.subheader("📊 Which factors influenced this prediction?")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(features, model.feature_importances_, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
# Added a permanent note at the bottom of the page
st.info("💡 **Note:** This is an educational AI engineering project. Predictions may not be entirely correct.")
st.markdown("🔗 [GitHub](https://github.com/jad-mrad) · Built with Python & Streamlit")