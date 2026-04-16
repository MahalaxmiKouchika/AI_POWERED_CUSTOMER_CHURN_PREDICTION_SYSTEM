import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .churn { background-color: #ffebee; color: #c62828; }
    .no-churn { background-color: #e8f5e9; color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load("customer_churn_model.pkl")
        training_columns = joblib.load("columns.pkl")
        return model, training_columns
    except:
        st.error("Model or columns file not found.")
        return None, None

model, training_columns = load_model()

st.title("Customer Churn Prediction")
st.markdown("Predict customer churn with Machine Learning")
st.markdown("---")

if model is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method",
                                     ["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0, 5.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, 50.0)

    st.markdown("---")

    if st.button("Predict Churn"):

        # 🔹 Create input dataframe
        input_data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
            'MultipleLines': multiple_lines, 'InternetService': internet_service,
            'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
            'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([input_data])

        # 🔹 Binary encoding
        input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

        # 🔹 One-hot encoding
        input_df = pd.get_dummies(input_df)

        # 🔹 Align with training columns
        input_df = input_df.reindex(columns=training_columns, fill_value=0)

        # 🔥 FINAL UNIVERSAL FIX (no more errors)
        input_df = input_df.infer_objects()
        input_df = input_df.apply(pd.to_numeric, errors='coerce')
        input_df = input_df.fillna(0)

        # 🔹 Prediction
        # 🔥 FORCE NUMERIC (fix error)
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]


        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box churn">WILL CHURN</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box no-churn">WILL STAY</div>', unsafe_allow_html=True)

        with col2:
            st.metric("Confidence", f"{max(proba) * 100:.1f}%")

        with col3:
            risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
            st.metric("Risk Level", risk)

        fig = go.Figure(data=[
            go.Bar(name='No Churn', x=['Probability'], y=[proba[0]]),
            go.Bar(name='Churn', x=['Probability'], y=[proba[1]])
        ])
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.warning("At Risk: Offer retention strategies")
        else:
            st.success("Low Risk: Customer likely to stay")

else:
    st.error("Model not loaded properly.")