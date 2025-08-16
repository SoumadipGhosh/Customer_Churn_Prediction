# Customer Churn Prediction

A machine learning web application for predicting customer churn using advanced analytics and interactive visualizations.

## 🌐 Live Demo

**Website:** [https://customer-churn-prediction-7luv.onrender.com/](https://customer-churn-prediction-7luv.onrender.com/)

## 📋 Overview

This application provides businesses with the ability to predict which customers are likely to churn (stop using their services) based on historical data and customer behavior patterns. The tool helps organizations proactively identify at-risk customers and take preventive measures to improve customer retention.

## ✨ Features

- **Interactive Web Interface**: User-friendly dashboard built with Streamlit
- **Real-time Predictions**: Get instant churn probability scores for individual customers
- **Data Visualization**: Interactive charts and graphs showing churn patterns and trends
- **Model Performance Metrics**: View accuracy, precision, recall, and other model statistics
- **Batch Predictions**: Upload CSV files for bulk customer churn analysis
- **Feature Importance**: Understand which factors most influence churn predictions
- **Customer Segmentation**: Analyze different customer segments and their churn rates

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Deployment**: Render

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 📊 How to Use

1. **Access the Application**: Visit the live demo at [https://customer-churn-prediction-7luv.onrender.com/](https://customer-churn-prediction-7luv.onrender.com/)

2. **Single Customer Prediction**:
   - Enter customer details in the sidebar
   - View real-time churn probability
   - Analyze feature contributions

3. **Batch Predictions**:
   - Upload a CSV file with customer data
   - Download results with churn probabilities
   - View summary statistics

4. **Data Exploration**:
   - Explore historical churn patterns
   - View feature distributions
   - Analyze correlation matrices

## 📈 Model Information

The application uses machine learning algorithms trained on historical customer data to predict churn probability. Key features typically include:

- Customer demographics
- Service usage patterns
- Billing information
- Customer service interactions
- Contract details
- Payment history

## 📋 Input Data Format

For batch predictions, ensure your CSV file contains the following columns:
- Customer ID
- Demographic information (age, gender, location)
- Service details (plan type, duration, usage)
- Financial data (monthly charges, total charges)
- Support interactions (calls, complaints)

## 🎯 Business Impact

- **Proactive Customer Retention**: Identify at-risk customers before they churn
- **Cost Reduction**: Reduce acquisition costs by retaining existing customers
- **Revenue Protection**: Maintain revenue streams through better customer management
- **Strategic Insights**: Understand key drivers of customer satisfaction

## 🔧 Configuration

The application can be customized by modifying:
- Model parameters in `config.py`
- Feature engineering in `preprocessing.py`
- Visualization themes in `visualizations.py`

## 📝 API Documentation

### Prediction Endpoint
- **URL**: `/predict`
- **Method**: POST
- **Input**: JSON with customer features
- **Output**: Churn probability and confidence score

### Health Check
- **URL**: `/health`
- **Method**: GET
- **Output**: Application status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or support:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation wiki

## 🚀 Deployment

The application is deployed on Render with automatic deployments from the main branch. Environment variables and secrets are managed through Render's dashboard.

## 📊 Performance Metrics

- **Model Accuracy**: ~85-90%
- **Precision**: ~80-85%
- **Recall**: ~75-80%
- **F1-Score**: ~77-82%

*Metrics may vary based on the dataset and model version*

## 🔄 Updates and Maintenance

- Regular model retraining with new data
- Performance monitoring and optimization
- Security updates and dependency management
- Feature enhancements based on user feedback

---

**Live Application**: [https://customer-churn-prediction-7luv.onrender.com/](https://customer-churn-prediction-7luv.onrender.com/)

*Built with ❤️ for better customer retention strategies*
