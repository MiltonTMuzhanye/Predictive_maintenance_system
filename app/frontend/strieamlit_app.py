"""
Streamlit Frontend Application

Interactive dashboard for predictive maintenance system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import numpy as np
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .status-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .status-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .status-low {
        color: #00cc66;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Predictive Maintenance")
st.sidebar.markdown("---")

# API Configuration
api_url = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000/api/v1",
    help="FastAPI endpoint URL"
)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predictions", "Customer Health", "Model Monitoring", "Reports"],
    index=0
)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'health_data' not in st.session_state:
    st.session_state.health_data = []

# Dashboard Page
if page == "Dashboard":
    st.markdown('<h1 class="main-header">📊 Predictive Maintenance Dashboard</h1>', unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">4,339</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">At Risk Customers</div>
            <div class="metric-value status-medium">1,024</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">High Risk</div>
            <div class="metric-value status-high">342</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Avg Health Score</div>
            <div class="metric-value">0.72</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Distribution")
        # Sample data
        churn_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Count': [2973, 682, 342]
        })
        fig = px.pie(churn_data, values='Count', names='Risk Level', 
                     color='Risk Level',
                     color_discrete_map={'Low': '#00cc66', 'Medium': '#ffa500', 'High': '#ff4b4b'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Health Score Distribution")
        # Generate sample data
        health_scores = np.random.normal(0.7, 0.15, 1000).clip(0, 1)
        fig = px.histogram(x=health_scores, nbins=30, title="Health Score Distribution")
        fig.update_layout(height=400, xaxis_title="Health Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    
    # Recent Activity
    st.subheader("Recent Predictions")
    
    # Sample prediction data
    recent_data = pd.DataFrame({
        'Timestamp': pd.date_range(end=datetime.now(), periods=10, freq='H'),
        'Customer ID': [f'CUST_{i:05d}' for i in range(1001, 1011)],
        'Churn Probability': np.random.uniform(0.1, 0.9, 10),
        'Risk Level': np.random.choice(['Low', 'Medium', 'High'], 10, p=[0.5, 0.3, 0.2]),
        'Health Score': np.random.uniform(0.2, 0.95, 10)
    })
    
    # Color coding for risk level
    def color_risk(val):
        if val == 'High':
            return 'color: #ff4b4b'
        elif val == 'Medium':
            return 'color: #ffa500'
        else:
            return 'color: #00cc66'
            
    st.dataframe(
        recent_data.style.applymap(color_risk, subset=['Risk Level']),
        use_container_width=True
    )

# Predictions Page
elif page == "Predictions":
    st.markdown('<h1 class="main-header">🎯 Customer Predictions</h1>', unsafe_allow_html=True)
    
    # Input form
    with st.expander("Make a Prediction", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.text_input("Customer ID", "CUST_12345")
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=5)
            frequency = st.number_input("Frequency (purchases in last 30 days)", min_value=0, value=10)
            
        with col2:
            monetary = st.number_input("Monetary (total spend)", min_value=0, value=500.0)
            avg_order_value = st.number_input("Average Order Value", min_value=0, value=25.0)
            days_since = st.number_input("Days Since Last Purchase", min_value=0, value=3)
            
        if st.button("Predict", type="primary"):
            # Prepare request
            request_data = {
                "customer_id": customer_id,
                "recency": recency,
                "frequency": frequency,
                "monetary": monetary,
                "avg_order_value": avg_order_value,
                "days_since_last_purchase": days_since
            }
            
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.predictions.append(result)
                    
                    # Display result
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Churn Probability",
                            f"{result['probability']:.2%}",
                            delta_color="inverse"
                        )
                        
                    with col2:
                        st.metric(
                            "Risk Level",
                            result['risk_level'].upper(),
                            delta_color="off"
                        )
                        
                    with col3:
                        st.metric(
                            "Health Score",
                            f"{result['health_score']:.2%}",
                            delta_color="normal"
                        )
                        
                    with col4:
                        st.metric(
                            "Health Status",
                            result['health_status'].upper(),
                            delta_color="off"
                        )
                        
                    # Recommendations
                    st.info(f"""
                    📋 **Recommendations:**
                    - Customer ID: {result['customer_id']}
                    - Risk Level: {result['risk_level'].upper()}
                    - Suggested Action: {'Immediate attention needed' if result['risk_level'] == 'high' 
                                        else 'Monitor closely' if result['risk_level'] == 'medium' 
                                        else 'Maintain engagement'}
                    """)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                
    # Prediction history
    if st.session_state.predictions:
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(history_df, use_container_width=True)
        
        # Download button
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Customer Health Page
elif page == "Customer Health":
    st.markdown('<h1 class="main-header">💊 Customer Health Assessment</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Health Metrics")
        
        # Sample health metrics
        metrics = {
            "Overall Health": 0.72,
            "Engagement Score": 0.65,
            "Purchase Frequency": 0.80,
            "Order Value Trend": 0.70,
            "Retention Likelihood": 0.75
        }
        
        for metric, value in metrics.items():
            st.metric(
                metric,
                f"{value:.1%}",
                delta=f"{'↑' if value > 0.7 else '↓'} {(value - 0.5):.1%}" if value != 0.72 else None
            )
            
    with col2:
        st.subheader("Health Score Gauge")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 72,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Customer Health Score"},
            delta = {'reference': 65, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "red"},
                    {'range': [33, 66], 'color': "orange"},
                    {'range': [66, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    
    # Health trends
    st.subheader("Health Score Trends")
    
    # Generate sample trend data
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='W')
    health_trends = pd.DataFrame({
        'Date': dates,
        'High Risk': np.random.uniform(20, 40, len(dates)),
        'Medium Risk': np.random.uniform(30, 50, len(dates)),
        'Low Risk': np.random.uniform(40, 60, len(dates))
    })
    
    fig = px.area(health_trends, x='Date', y=['High Risk', 'Medium Risk', 'Low Risk'],
                  title="Customer Risk Distribution Over Time")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Model Monitoring Page
elif page == "Model Monitoring":
    st.markdown('<h1 class="main-header">📈 Model Monitoring</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        performance = {
            "Accuracy": 0.85,
            "Precision": 0.82,
            "Recall": 0.79,
            "F1 Score": 0.80,
            "AUC-ROC": 0.88
        }
        
        for metric, value in performance.items():
            st.metric(
                metric,
                f"{value:.2%}",
                delta=f"{'+' if value > 0.8 else '-'}{value - 0.8:.1%}"
            )
            
    with col2:
        st.subheader("Feature Importance")
        
        # Sample feature importance
        features = {
            'Recency': 0.25,
            'Frequency': 0.20,
            'Monetary': 0.18,
            'Avg Order Value': 0.15,
            'Days Since Purchase': 0.12,
            'Purchase History': 0.10
        }
        
        fig = px.bar(
            x=list(features.values()),
            y=list(features.keys()),
            orientation='h',
            title="Feature Importance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    
    st.subheader("Data Drift Detection")
    
    # Sample drift data
    drift_data = pd.DataFrame({
        'Feature': ['Recency', 'Frequency', 'Monetary', 'Avg Order Value', 'Days Since Purchase'],
        'Drift Score': [0.05, 0.12, 0.08, 0.03, 0.15],
        'Alert': ['No', 'Warning', 'No', 'No', 'Warning']
    })
    
    st.dataframe(drift_data, use_container_width=True)

# Reports Page
elif page == "Reports":
    st.markdown('<h1 class="main-header">📊 Reports & Analytics</h1>', unsafe_allow_html=True)
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Customer Health Summary", "Churn Analysis", "Risk Assessment", "Maintenance Recommendations"]
    )
    
    if report_type == "Customer Health Summary":
        st.subheader("Customer Health Summary Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Health Score Distribution
            
            - **Excellent**: 1,234 (28.4%)
            - **Good**: 1,876 (43.2%)
            - **Fair**: 987 (22.7%)
            - **Poor**: 242 (5.6%)
            """)
            
        with col2:
            st.markdown("""
            ### Key Insights
            
            - ✅ Average health score: **0.72**
            - 📈 Health score trend: **+2.3%** this quarter
            - ⚠️ High risk customers: **342**
            - 📉 Churn rate: **3.2%** (decreasing)
            """)
            
    elif report_type == "Churn Analysis":
        st.subheader("Churn Analysis Report")
        
        # Sample churn data
        churn_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'At Risk': [120, 115, 110, 108, 105, 102],
            'Churned': [25, 22, 20, 18, 16, 15]
        })
        
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=churn_data['Month'], y=churn_data['At Risk'],
                                 mode='lines+markers', name='At Risk'))
        fig.add_trace(go.Scatter(x=churn_data['Month'], y=churn_data['Churned'],
                                 mode='lines+markers', name='Churned'))
        fig.update_layout(title="Churn Trends", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    elif report_type == "Risk Assessment":
        st.subheader("Risk Assessment Report")
        
        st.markdown("""
        ### Top Risk Factors
        
        1. **High Recency**: Customers with 30+ days since last purchase
        2. **Low Frequency**: Customers with <5 purchases in last 90 days
        3. **Declining Monetary Value**: Decreasing average order value
        4. **Low Engagement**: Reduced interaction with marketing
        """)
        
    else:  # Maintenance Recommendations
        st.subheader("Maintenance Recommendations")
        
        st.markdown("""
        ### Actionable Recommendations
        
        #### High Risk Customers (342)
        - 📞 Immediate outreach required
        - 🎁 Loyalty program offers
        - 📧 Personalized re-engagement campaigns
        
        #### Medium Risk Customers (682)
        - 📱 Regular check-ins
        - 💰 Targeted discounts
        - 📊 Monitor engagement metrics
        
        #### Low Risk Customers (2,973)
        - 📨 Maintain communication
        - 🏆 Recognition programs
        - 📈 Continue monitoring
        """)

if __name__ == "__main__":
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            Predictive Maintenance System v1.0.0 | 
            Powered by XGBoost & Deep Learning
        </div>
        """,
        unsafe_allow_html=True
    )