import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Prediction", layout="wide")

# 🔥 FULL UI STYLING (Buttons + Tabs)
st.markdown("""
    <style>

    /* 🔴 Predict Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B, #ff7b7b);
        color: white;
        font-weight: bold;
        padding: 0.6rem;
        border-radius: 12px;
        border: none;
        font-size: 16px;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #ff2e2e, #ff6b6b);
        transform: scale(1.03);
        transition: 0.3s;
    }

    /* 🔥 Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 25px;
        border-radius: 12px;
        background-color: #1f2a40;
        color: white;
        font-size: 16px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #FF4B4B, #ff7b7b);
        color: white;
        box-shadow: 0 4px 15px rgba(255,75,75,0.4);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2e3b55;
        color: #FF4B4B;
        transform: scale(1.05);
        transition: 0.3s;
    }

    </style>
""", unsafe_allow_html=True)


# 🔹 Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("customer_churn_model.pkl")
        training_columns = joblib.load("columns.pkl")
        return model, training_columns
    except:
        st.error("Model or columns file not found.")
        return None, None


# 🔥 AI Insight Function
def generate_ai_insight(data):
    insights = []

    if data['Contract'] == 'Month-to-month':
        insights.append("📉 Offer long-term contract")

    if data['MonthlyCharges'] > 80:
        insights.append("💰 Provide discount")

    if data['tenure'] < 12:
        insights.append("🤝 Improve onboarding")

    if data['TechSupport'] == 'No':
        insights.append("🛠️ Offer tech support")

    if data['InternetService'] == 'Fiber optic':
        insights.append("📡 Check service quality")

    if data['OnlineSecurity'] == 'No':
        insights.append("🔐 Add security features")

    if len(insights) == 0:
        return ["✅ Customer stable"]

    return insights


model, training_columns = load_model()

# 🔥 ATTRACTIVE TABS
tab1, tab2 = st.tabs([
    "Predict Customer",
    "Business Dashboard"
])


# =========================
# 🔮 PREDICTION TAB
# =========================
with tab1:

    st.title("Customer Churn Prediction")
    st.markdown("Predict customer churn with Machine Learning")
    st.markdown("---")

    if model is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
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
            tenure = st.slider("Tenure", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method",
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges (₹)", 0.0, 150.0, 50.0)
            total_charges = st.number_input("Total Charges (₹)", 0.0, 10000.0, monthly_charges * tenure)

        st.markdown("---")

        if st.button("🚀 Predict Now"):

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
            input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})

            for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=training_columns, fill_value=0)
            input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.markdown("---")

            if prediction == 1:
                st.error("🚨 WILL CHURN")
            else:
                st.success("✅ WILL STAY")

            st.metric("Confidence", f"{max(proba)*100:.2f}%")

            fig = go.Figure(data=[
                go.Bar(name='No Churn', x=['Probability'], y=[proba[0]]),
                go.Bar(name='Churn', x=['Probability'], y=[proba[1]])
            ])
            st.plotly_chart(fig, use_container_width=True)

            # 🤖 AI Insights
            st.markdown("### 🤖 AI Insights")
            insights = generate_ai_insight(input_data)
            for i in insights:
                st.write("•", i)

    else:
        st.error("Model not loaded properly.")


# =========================
# 📊 ADMIN DASHBOARD
# =========================
with tab2:

    st.title("📊 Business Dashboard")

    try:
        df = pd.read_csv("Telco-Customer-Churn.csv")

        st.subheader("Dataset Preview")
        st.write(df.head())

        churn_counts = df['Churn'].value_counts()
        fig1 = go.Figure(data=[go.Pie(labels=churn_counts.index, values=churn_counts.values)])
        st.plotly_chart(fig1)

        contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig2 = go.Figure()
        for val in contract_churn['Churn'].unique():
            temp = contract_churn[contract_churn['Churn'] == val]
            fig2.add_bar(x=temp['Contract'], y=temp['Count'], name=str(val))
        st.plotly_chart(fig2)

    except:
        st.error("⚠️ Add customer_churn_data.csv file")