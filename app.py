import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="ChurnPredict Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal, reliable CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .prediction-high-risk {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-low-risk {
        background: linear-gradient(90deg, #26d0ce 0%, #1dd1a1 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model_data, encoders
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def create_hero_section():
    """Create hero section with native Streamlit styling"""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 ChurnPredict Pro</h1>
        <h3>Advanced Customer Retention Analytics Platform</h3>
        <p>Predict customer churn with machine learning precision</p>
    </div>
    """, unsafe_allow_html=True)

def create_input_form(encoders):
    """Create input form using native Streamlit components"""
    
    st.header("📊 Customer Data Input")
    
    # Demographics Section
    st.subheader("👤 Customer Demographics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col3:
        partner = st.selectbox("Has Partner", ["No", "Yes"])
    with col4:
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    st.divider()
    
    # Services Section
    st.subheader("📱 Service Subscriptions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    
    with col3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    st.divider()
    
    # Contract & Billing Section
    st.subheader("💼 Contract & Billing")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    
    with col2:
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    with col3:
        tenure = st.slider("Tenure (months)", 0, 72, 12, 
                          help="Number of months the customer has been with the company")
    
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0)
    
    # Convert to model format
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
        df = pd.DataFrame([input_data])
        
        for column in encoders.keys():
            if column in df.columns:
                try:
                    df[column] = encoders[column].transform(df[column])
                except ValueError:
                    df[column] = encoders[column].transform([encoders[column].classes_[0]])
        
        prediction = model_data['model'].predict(df)[0]
        prediction_proba = model_data['model'].predict_proba(df)[0]
        feature_importance = model_data['model'].feature_importances_
        
        return prediction, prediction_proba, feature_importance
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        return None, None, None

def create_prediction_visualization(prediction_proba):
    """Create prediction visualization using Plotly"""
    churn_prob = prediction_proba[1] * 100
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = churn_prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Risk Score (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 50], 'color': "yellow"},
                {'range': [50, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70}}))
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def display_prediction_results(prediction, prediction_proba):
    """Display prediction results using native Streamlit components"""
    
    st.header("🎯 Prediction Results")
    
    churn_prob = prediction_proba[1] * 100
    stay_prob = prediction_proba[0] * 100
    
    # Main prediction result
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-high-risk">
            <h2>⚠️ HIGH CHURN RISK</h2>
            <h3>Churn Probability: {churn_prob:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-low-risk">
            <h2>✅ LOW CHURN RISK</h2>
            <h3>Retention Probability: {stay_prob:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics using native Streamlit
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Retention Probability", f"{stay_prob:.1f}%")
    with col2:
        st.metric("Churn Probability", f"{churn_prob:.1f}%")
    with col3:
        confidence = max(stay_prob, churn_prob)
        st.metric("Model Confidence", f"{confidence:.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Gauge")
        fig = create_prediction_visualization(prediction_proba)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Probability Breakdown")
        fig2 = go.Figure(data=[
            go.Bar(name='Probability', 
                   x=['Will Stay', 'Will Churn'], 
                   y=[stay_prob, churn_prob],
                   marker_color=['#1dd1a1', '#ee5a24'])
        ])
        fig2.update_layout(height=400, showlegend=False, yaxis_title="Probability (%)")
        st.plotly_chart(fig2, use_container_width=True)

def show_recommendations(prediction, prediction_proba):
    """Show actionable recommendations"""
    
    st.header("💡 Actionable Recommendations")
    
    churn_prob = prediction_proba[1] * 100
    
    if prediction == 1:  # High churn risk
        if churn_prob >= 80:
            st.error("🚨 IMMEDIATE ACTION REQUIRED")
        elif churn_prob >= 60:
            st.warning("⚠️ HIGH PRIORITY")
        else:
            st.info("📋 MODERATE PRIORITY")
        
        st.subheader("🎯 Immediate Actions:")
        st.write("• Schedule personal outreach within 24-48 hours")
        st.write("• Conduct service satisfaction survey")
        st.write("• Prepare retention offer package")
        st.write("• Address any service issues proactively")
        
        st.subheader("📊 Strategic Actions:")
        st.write("• Offer longer-term contract with better rates")
        st.write("• Suggest service bundling for added value")
        st.write("• Provide flexible payment options")
        st.write("• Enroll in loyalty program")
        
        st.markdown("""
        <div class="warning-box">
            <strong>⏰ Timeline:</strong> Implement interventions within 1-2 weeks for maximum effectiveness.
        </div>
        """, unsafe_allow_html=True)
    
    else:  # Low churn risk
        st.success("✅ RETENTION OPPORTUNITIES")
        
        st.subheader("🔄 Growth Strategies:")
        st.write("• Explore upselling opportunities")
        st.write("• Offer complementary services")
        st.write("• Leverage satisfaction for referrals")
        st.write("• Collect feedback on satisfaction drivers")
        
        st.subheader("🛡️ Preventive Measures:")
        st.write("• Schedule quarterly check-ins")
        st.write("• Continue service optimization")
        st.write("• Provide early access to new features")
        st.write("• Implement appreciation programs")
        
        st.markdown("""
        <div class="success-box">
            <strong>🎯 Focus:</strong> Maintain satisfaction while exploring growth opportunities.
        </div>
        """, unsafe_allow_html=True)

def show_customer_profile(input_data, prediction):
    """Show customer profile summary"""
    
    st.header("📋 Customer Profile Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profile Highlights")
        tenure_category = "New" if input_data['tenure'] < 12 else "Established" if input_data['tenure'] < 36 else "Long-term"
        value_category = "Budget" if input_data['MonthlyCharges'] < 35 else "Standard" if input_data['MonthlyCharges'] < 75 else "Premium"
        
        st.write(f"**Customer Segment:** {tenure_category} {value_category}")
        st.write(f"**Service Tenure:** {input_data['tenure']} months")
        st.write(f"**Monthly Value:** ${input_data['MonthlyCharges']:.2f}")
        st.write(f"**Contract Type:** {input_data['Contract']}")
        st.write(f"**Internet Service:** {input_data['InternetService']}")
    
    with col2:
        st.subheader("⚡ Quick Actions")
        if prediction == 1:
            st.write("📞 Schedule retention call")
            st.write("💰 Prepare retention offer")
            st.write("📊 Review service usage")
            st.write("🎯 Create action plan")
            st.write("📅 Set follow-up reminder")
        else:
            st.write("📈 Explore upsell opportunities")
            st.write("🌟 Enroll in loyalty program")
            st.write("💬 Collect satisfaction feedback")
            st.write("🔄 Schedule regular check-in")
            st.write("🎁 Consider appreciation gesture")

def show_model_info():
    """Show model documentation using native Streamlit components"""
    
    st.header("📚 Model Documentation")
    
    tab1, tab2, tab3 = st.tabs(["🤖 Model Overview", "🛠️ Methodology", "📊 Feature Importance"])
    
    with tab1:
        st.subheader("Random Forest Classifier")
        st.write("""
        This application uses a **Random Forest Classifier**, an ensemble machine learning 
        algorithm that combines multiple decision trees to make accurate predictions.
        """)
        
        st.markdown("""
        <div class="info-box">
            <strong>Why Random Forest?</strong><br>
            • High accuracy and reliability<br>
            • Handles mixed data types well<br>
            • Provides feature importance rankings<br>
            • Resistant to overfitting<br>
            • Handles missing values effectively
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Key Characteristics")
        st.write("**Algorithm Type:** Supervised Learning (Classification)")
        st.write("**Method:** Bootstrap Aggregating (Bagging)")
        st.write("**Output:** Binary classification (Churn: Yes/No)")
        st.write("**Probability:** Based on tree vote distribution")
    
    with tab2:
        st.subheader("Data Processing Pipeline")
        
        st.write("**1. Data Cleaning:** Handle missing values and inconsistencies")
        st.write("**2. Feature Engineering:** Transform categorical variables")
        st.write("**3. Class Balancing:** Apply SMOTE for balanced training")
        st.write("**4. Model Training:** Train with optimized parameters")
        st.write("**5. Validation:** Cross-validation and evaluation")
        
        st.subheader("Model Interpretation")
        st.write("""
        The model calculates probabilities based on the proportion of trees voting 
        for each outcome. Higher confidence indicates stronger agreement among trees.
        """)
        
        st.markdown("""
        <div class="info-box">
            <strong>Confidence Levels:</strong><br>
            • 50-60%: Monitor closely<br>
            • 60-80%: Moderate confidence<br>
            • 80%+: High confidence
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Understanding Feature Importance")
        st.write("""
        Feature importance scores show how much each input contributes to prediction accuracy.
        """)
        
        st.write("**Top Contributing Factors:**")
        st.write("1. **Monthly Charges** - High Impact")
        st.write("2. **Contract Type** - High Impact")
        st.write("3. **Tenure** - High Impact")
        st.write("4. **Internet Service** - Medium Impact")
        st.write("5. **Payment Method** - Medium Impact")
        
        st.markdown("""
        <div class="info-box">
            <strong>Key Insights:</strong><br>
            • Higher charges often increase churn risk<br>
            • Longer contracts reduce churn probability<br>
            • Longer tenure indicates loyalty<br>
            • Service type affects retention patterns
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Hero section
    create_hero_section()
    
    # Load model
    model_data, encoders = load_model_and_encoders()
    
    if model_data is None or encoders is None:
        st.error("❌ **System Status: Offline**")
        st.write("Required files:")
        st.write("• `customer_churn_model.pkl`")
        st.write("• `encoders.pkl`")
        st.info("Run the model training script to generate these files.")
        return
    
    # Success message
    st.success("✅ **System Status: Online** - Model ready for predictions")
    
    # Main tabs
    tab1, tab2 = st.tabs(["🎯 Prediction Interface", "📚 Model Documentation"])
    
    with tab1:
        # Input form
        input_data = create_input_form(encoders)
        
        # Prediction button
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 **Generate Churn Prediction**", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing customer data..."):
                    prediction, prediction_proba, feature_importance = make_prediction(input_data, model_data, encoders)
                    
                    if prediction is not None:
                        # Show results
                        display_prediction_results(prediction, prediction_proba)
                        
                        st.divider()
                        
                        # Show recommendations
                        show_recommendations(prediction, prediction_proba)
                        
                        st.divider()
                        
                        # Show customer profile
                        show_customer_profile(input_data, prediction)
    
    with tab2:
        show_model_info()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
        <h3>🎯 ChurnPredict Pro</h3>
        <p><strong>Advanced Customer Retention Analytics</strong></p>
        <p>Powered by Machine Learning • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()