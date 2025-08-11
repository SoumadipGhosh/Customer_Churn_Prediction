import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.prediction-box {
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
    margin: 1rem 0;
}
.churn-yes {
    background-color: #ffebee;
    color: #c62828;
    border: 2px solid #c62828;
}
.churn-no {
    background-color: #e8f5e8;
    color: #2e7d32;
    border: 2px solid #2e7d32;
}
.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        # Load model
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Load encoders
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        
        return model_data, encoders
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the training script first. Error: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

def create_input_form(encoders):
    """Create the input form for customer data"""
    
    st.markdown('<h2 class="sub-header">📝 Customer Information</h2>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
    with col2:
        st.markdown("**📞 Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
    with col3:
        st.markdown("**💼 Additional Services**")
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    # Contract and payment info
    st.markdown("**💳 Contract & Payment**")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        
    with col5:
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
    with col6:
        tenure = st.slider("Tenure (months)", 0, 72, 12, help="Number of months the customer has stayed with the company")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0)
    
    # Convert inputs to the format expected by the model
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    return input_data

def make_prediction(input_data, model_data, encoders):
    """Make prediction using the trained model"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Apply the same preprocessing as training
        for column in encoders.keys():
            if column in df.columns:
                # Handle new categories that weren't in training data
                try:
                    df[column] = encoders[column].transform(df[column])
                except ValueError:
                    # If unknown category, use the most frequent category
                    df[column] = encoders[column].transform([encoders[column].classes_[0]])
        
        # Make prediction
        prediction = model_data['model'].predict(df)[0]
        prediction_proba = model_data['model'].predict_proba(df)[0]
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def display_prediction(prediction, prediction_proba):
    """Display the prediction results"""
    st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
    
    if prediction is not None:
        churn_probability = prediction_proba[1] * 100
        no_churn_probability = prediction_proba[0] * 100
        
        # Main prediction
        if prediction == 1:
            st.markdown(
                f'<div class="prediction-box churn-yes">⚠️ HIGH RISK: Customer likely to churn<br>Probability: {churn_probability:.1f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-box churn-no">✅ LOW RISK: Customer likely to stay<br>Probability: {no_churn_probability:.1f}%</div>',
                unsafe_allow_html=True
            )
        
        # Detailed probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of Staying", f"{no_churn_probability:.1f}%", 
                     delta=f"{no_churn_probability - 50:.1f}%" if no_churn_probability != 50 else None)
        with col2:
            st.metric("Probability of Churning", f"{churn_probability:.1f}%",
                     delta=f"{churn_probability - 50:.1f}%" if churn_probability != 50 else None,
                     delta_color="inverse")
        
        # Progress bars
        st.markdown("**Confidence Breakdown:**")
        st.progress(no_churn_probability / 100, text=f"Will Stay: {no_churn_probability:.1f}%")
        st.progress(churn_probability / 100, text=f"Will Churn: {churn_probability:.1f}%")
        
        # Recommendations
        st.markdown('<h3 class="sub-header">💡 Recommendations</h3>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            <div class="info-box">
            <h4>🚨 Retention Strategies:</h4>
            <ul>
                <li>Contact customer proactively to understand concerns</li>
                <li>Offer personalized discounts or promotions</li>
                <li>Provide additional customer support</li>
                <li>Consider contract renegotiation with better terms</li>
                <li>Enhance service quality based on customer feedback</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <h4>😊 Customer Retention:</h4>
            <ul>
                <li>Customer shows good retention indicators</li>
                <li>Consider upselling additional services</li>
                <li>Maintain current service quality</li>
                <li>Regular check-ins to ensure continued satisfaction</li>
                <li>Loyalty program enrollment opportunity</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and encoders
    model_data, encoders = load_model_and_encoders()
    
    if model_data is None or encoders is None:
        st.error("❌ Unable to load model files. Please ensure 'customer_churn_model.pkl' and 'encoders.pkl' are in the same directory.")
        st.info("💡 Run the training script first to generate these files.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 📋 About This App")
        st.info("""
        This application predicts whether a telecom customer is likely to churn (leave) based on their:
        - Demographics
        - Service subscriptions
        - Contract details
        - Payment information
        """)
        
        st.markdown("### 🔧 Model Information")
        st.write(f"**Algorithm:** Random Forest Classifier")
        st.write(f"**Features:** {len(model_data['features_names'])}")
        st.write(f"**Data Processing:** SMOTE for class balancing")
        
        st.markdown("### 📈 Prediction Scale")
        st.write("- **0-30%:** Very Low Risk")
        st.write("- **30-50%:** Low Risk") 
        st.write("- **50-70%:** Moderate Risk")
        st.write("- **70-90%:** High Risk")
        st.write("- **90-100%:** Very High Risk")
    
    # Main input form
    input_data = create_input_form(encoders)
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Predict Customer Churn", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer data..."):
                prediction, prediction_proba = make_prediction(input_data, model_data, encoders)
                display_prediction(prediction, prediction_proba)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; padding: 2rem;'>
        <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
        <p><em>Helping businesses reduce customer churn through predictive analytics</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()