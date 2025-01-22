import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from jira import JIRA
import base64
from datetime import datetime, timedelta
import io
import json
import openpyxl
import os

# Page config
st.set_page_config(
    page_title="Test Case Prioritization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-header {
        color: #1E88E5;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .download-button {
        background-color: #1976D2;
        border: none;
        color: #FFE57F !important;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        font-weight: 600;
        margin: 4px 2px;
        border-radius: 4px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        width: 100%;
        cursor: pointer;
    }
    .download-button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background-color: #1976D2;
        color: #FFFFFF;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .nav-link {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        color: #424242;
        text-decoration: none;
        display: block;
        transition: all 0.3s;
    }
    .nav-link:hover {
        background-color: #e9ecef;
        color: #1E88E5;
    }
    .nav-link.active {
        background-color: #1E88E5;
        color: white;
    }
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for persistent data
if 'model' not in st.session_state:
    try:
        with open('model/test_case_priority_model.pkl', 'rb') as model_file:
            st.session_state.model = pickle.load(model_file)
    except Exception as e:
        st.error("Please run train_new_model.py first to generate the model!")
        st.stop()

if 'history' not in st.session_state:
    st.session_state.history = []

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

# Sidebar navigation with improved UI
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/test-lab.png", width=100)
    st.title("Navigation")
    
    # Navigation menu with better styling
    pages = ["Welcome", "Test Case Analysis", "Historical Trends"]
    page = st.radio("", pages, label_visibility="collapsed")
    
    # JIRA Configuration (always visible)
    st.markdown("---")
    st.header("JIRA Integration")
    with st.expander("Configure JIRA", expanded=True):
        jira_url = st.text_input("JIRA URL", placeholder="https://your-domain.atlassian.net")
        jira_email = st.text_input("Email")
        jira_api_token = st.text_input("API Token", type="password")
        jira_project = st.text_input("Project Key")

# Initialize session state
if 'jira_url' not in st.session_state:
    st.session_state.jira_url = jira_url
if 'jira_email' not in st.session_state:
    st.session_state.jira_email = jira_email
if 'jira_api_token' not in st.session_state:
    st.session_state.jira_api_token = jira_api_token
if 'jira_project' not in st.session_state:
    st.session_state.jira_project = jira_project

def connect_to_jira():
    if st.session_state.jira_url and st.session_state.jira_email and st.session_state.jira_api_token:
        try:
            return JIRA(
                server=st.session_state.jira_url,
                basic_auth=(st.session_state.jira_email, st.session_state.jira_api_token)
            )
        except Exception as e:
            st.error(f"Failed to connect to JIRA: {str(e)}")
    return None

def create_jira_issue(jira, test_case, priority):
    if jira and st.session_state.jira_project:
        try:
            issue_dict = {
                'project': {'key': st.session_state.jira_project},
                'summary': f'High Priority Test Case: {test_case}',
                'description': f'Test case requires attention.\nPriority Score: {priority}\nDetected by AI Test Case Prioritization System',
                'issuetype': {'name': 'Task'},
                'priority': {'name': 'High'}
            }
            jira.create_issue(fields=issue_dict)
            return True
        except Exception as e:
            st.error(f"Failed to create JIRA issue: {str(e)}")
    return False

def calculate_risk_score(row):
    # Convert severity to numeric value
    severity_map = {'low': 1, 'medium': 2, 'high': 3}
    severity_numeric = severity_map.get(str(row['defect_severity']).lower(), 1)
    
    # Normalize severity to 0-1 range
    severity_weight = (severity_numeric - 1) / 2.0
    
    # Calculate coverage weight (0-1 range)
    coverage_weight = 1 - (float(row['code_coverage']) / 100)
    
    # Calculate execution weight
    execution_weight = 1.0 if str(row['execution_result']).lower() == 'fail' else 0.2
    
    # Calculate final risk score
    risk_score = (
        severity_weight * 0.4 +
        coverage_weight * 0.3 +
        execution_weight * 0.3
    )
    return risk_score * 100

def detect_failure_patterns(df):
    patterns = []
    
    # Consecutive failures
    df['execution_numeric'] = df['execution_result'].map({'pass': 0, 'fail': 1})
    consecutive_failures = df[df['execution_numeric'] == 1].groupby(
        (df['execution_numeric'] != df['execution_numeric'].shift()).cumsum()
    ).size()
    
    if len(consecutive_failures) > 0:
        patterns.append(f"Found {len(consecutive_failures)} sequences of consecutive failures")
        
    # Failure rate by severity
    df_failures = df[df['execution_numeric'] == 1].groupby('defect_severity').size()
    df_total = df.groupby('defect_severity').size()
    failure_rate = (df_failures / df_total * 100).round(2)
    
    for severity, rate in failure_rate.items():
        patterns.append(f"{severity.capitalize()} severity tests have {rate}% failure rate")
    
    return patterns

def get_csv_download_link(df, filename="predicted_test_cases.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">Download {filename}</a>'
    return href

def export_to_excel(df, filename="test_case_analysis.xlsx"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Main data
        df.to_excel(writer, sheet_name='Test Cases', index=False)
        
        # Summary statistics
        summary = pd.DataFrame({
            'Metric': ['Total Test Cases', 'High Priority Cases', 'Average Risk Score'],
            'Value': [len(df), sum(df['priority']), df['risk_score'].mean()]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Failure analysis
        failure_analysis = df.groupby('defect_severity')['execution_result'].value_counts().unstack()
        failure_analysis.to_excel(writer, sheet_name='Failure Analysis')
        
    return buffer

def export_to_json(df):
    return df.to_json(orient='records', date_format='iso')

def export_to_html(df):
    html_content = """
    <html>
    <head>
        <title>Test Case Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
    """
    html_content += "<h1>Test Case Analysis Report</h1>"
    html_content += "<h2>Summary</h2>"
    html_content += f"<p>Total Test Cases: {len(df)}</p>"
    html_content += f"<p>High Priority Cases: {sum(df['priority'])}</p>"
    html_content += f"<p>Average Risk Score: {df['risk_score'].mean():.2f}</p>"
    html_content += "<h2>Test Cases</h2>"
    html_content += df.to_html(classes='table')
    html_content += "</body></html>"
    return html_content

# Main app content
if page == "Welcome":
    st.markdown('<h1 class="main-header">🎯 AI Test Case Prioritization System</h1>', unsafe_allow_html=True)
    
    # Show current analysis summary if available
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
        st.success("Current analysis loaded! You can proceed to Test Case Analysis to view details.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Test Cases", len(df))
        with col2:
            st.metric("High Priority Cases", sum(df['priority']))
        with col3:
            st.metric("Average Risk Score", f"{df['risk_score'].mean():.2f}")
    
    st.markdown("""
        <div class="card">
            <h2 class="feature-header">What is this?</h2>
            <p>An intelligent system that uses machine learning to help you prioritize your test cases effectively.
            It analyzes multiple factors including defect severity, code coverage, and execution history to identify
            high-priority test cases that need immediate attention.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="card">
                <h2 class="feature-header">Key Features</h2>
                <ul>
                    <li>🎯 Automatic test case prioritization</li>
                    <li>📊 Comprehensive risk assessment</li>
                    <li>🔍 Failure pattern detection</li>
                    <li>📈 Coverage analysis</li>
                    <li>📱 JIRA integration</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card">
                <h2 class="feature-header">How to Use</h2>
                <ol>
                    <li>Prepare your CSV file with test case data</li>
                    <li>Upload it in the "Test Case Analysis" page</li>
                    <li>View comprehensive analysis across different tabs</li>
                    <li>Export results in your preferred format</li>
                    <li>Optionally create JIRA issues for high-priority cases</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

elif page == "Test Case Analysis":
    st.markdown('<h1 class="main-header">Test Case Analysis</h1>', unsafe_allow_html=True)
    
    # File Upload Section
    st.subheader("Upload Test Cases Data")
    
    # Add file format instructions
    st.markdown("""
    ### File Format Instructions
    Please upload a CSV file with the following columns:
    - `test_case_id`: Unique identifier for each test case (e.g., TC001)
    - `execution_result`: Test execution outcome (pass/fail)
    - `defect_severity`: Severity level of defects (low/medium/high)
    - `code_coverage`: Code coverage percentage (numeric value between 50-100)
    
    Download our sample template to get started:
    """)
    
    # Add sample template download button
    sample_path = "data/samples/sample_test_cases_template.csv"
    with open(sample_path, "r") as f:
        csv_content = f.read()
    b64_csv = base64.b64encode(csv_content.encode()).decode()
    st.markdown(f"""
        <a href="data:file/csv;base64,{b64_csv}" download="test_cases_template.csv" 
           style="display: inline-block; padding: 0.5rem 1rem; 
                  background-color: #4CAF50; color: white; 
                  text-decoration: none; border-radius: 4px;
                  margin-bottom: 1rem;">
            📥 Download Template CSV
        </a>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Process data
    df = None
    if uploaded_file is None and st.session_state.current_data is None:
        st.info("No file uploaded. Using sample test cases for demonstration.")
        df = pd.read_csv("data/sample_test_cases.csv")
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_data = df  # Store in session state
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        df = st.session_state.current_data
    
    if df is not None:
        try:
            required_columns = ['test_case_id', 'execution_result', 'defect_severity', 'code_coverage']
            
            # Validate required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            
            # Clean and standardize the data
            df['execution_result'] = df['execution_result'].str.lower()
            df['defect_severity'] = df['defect_severity'].str.lower()
            df['code_coverage'] = pd.to_numeric(df['code_coverage'], errors='coerce')
            
            # Fill any missing values
            df['code_coverage'] = df['code_coverage'].fillna(df['code_coverage'].mean())
            df['execution_result'] = df['execution_result'].fillna('pass')
            df['defect_severity'] = df['defect_severity'].fillna('low')
            
            # Calculate risk scores
            df['risk_score'] = df.apply(calculate_risk_score, axis=1)
            
            # Make predictions
            df_processed = df.copy()
            df_processed['defect_severity'] = df_processed['defect_severity'].map({'low': 1, 'medium': 2, 'high': 3})
            df_processed['execution_result'] = df_processed['execution_result'].map({'pass': 0, 'fail': 1})
            
            features = ['defect_severity', 'code_coverage', 'execution_result']
            predictions = st.session_state.model.predict(df_processed[features])
            df['priority'] = predictions
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Coverage Analysis", "Failure Patterns", "Risk Assessment"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Test Cases", len(df))
                with col2:
                    st.metric("High Priority Cases", sum(predictions))
                with col3:
                    st.metric("Average Risk Score", f"{df['risk_score'].mean():.2f}")
                
                # Priority Distribution
                fig = px.pie(df, names='priority', title='Priority Distribution',
                            color_discrete_map={0: '#3498db', 1: '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                st.markdown(get_csv_download_link(df), unsafe_allow_html=True)
                
                # JIRA Integration
                if st.button("Create JIRA Issues for High Priority Cases"):
                    jira = connect_to_jira()
                    if jira:
                        high_priority_cases = df[df['priority'] == 1]
                        for _, case in high_priority_cases.iterrows():
                            if create_jira_issue(jira, case['test_case_id'], case['risk_score']):
                                st.success(f"Created JIRA issue for test case {case['test_case_id']}")
            
            with tab2:
                # Coverage Analysis
                st.subheader("Code Coverage Analysis")
                
                # Coverage distribution
                fig = px.histogram(df, x='code_coverage', nbins=20,
                                 title='Code Coverage Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Coverage by severity
                fig = px.box(df, x='defect_severity', y='code_coverage',
                            title='Code Coverage by Defect Severity')
                st.plotly_chart(fig, use_container_width=True)
                
                # Coverage vs Risk Score
                fig = px.scatter(df, x='code_coverage', y='risk_score',
                               color='priority', title='Code Coverage vs Risk Score')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Failure Pattern Analysis
                st.subheader("Failure Pattern Analysis")
                
                patterns = detect_failure_patterns(df)
                for pattern in patterns:
                    st.info(pattern)
                
                # Failure trend
                fig = px.line(df, x=df.index, y='execution_numeric',
                             title='Test Execution Result Trend')
                st.plotly_chart(fig, use_container_width=True)
                
                # Failure correlation matrix
                correlation = df[['code_coverage', 'execution_numeric', 'priority', 'risk_score']].corr()
                fig = px.imshow(correlation, title='Correlation Matrix')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Risk Assessment
                st.subheader("Risk Assessment")
                
                # Risk score distribution
                fig = px.histogram(df, x='risk_score', nbins=20,
                                 title='Risk Score Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors breakdown by severity
                risk_by_severity = df.groupby('defect_severity')['risk_score'].mean().reset_index()
                fig = px.bar(risk_by_severity, x='defect_severity', y='risk_score',
                            title='Average Risk Score by Severity')
                st.plotly_chart(fig, use_container_width=True)
                
                # High risk test cases
                st.subheader("High Risk Test Cases")
                high_risk_cases = df[df['risk_score'] > df['risk_score'].quantile(0.75)]
                st.dataframe(high_risk_cases)
            
            # Add export options
            st.markdown("### Export Results")
            st.markdown('<div style="padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin: 1rem 0;">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                csv = df.to_csv(index=False)
                b64_csv = base64.b64encode(csv.encode()).decode()
                st.markdown("""
                    <div style="text-align: center;">
                        <p style="margin-bottom: 0.5rem; color: #666;">Basic Data</p>
                        <a href="data:file/csv;base64,{}" download="test_cases_analysis.csv" class="download-button">
                            📥 Download CSV
                        </a>
                    </div>
                """.format(b64_csv), unsafe_allow_html=True)
            
            with col2:
                excel_buffer = export_to_excel(df)
                b64_excel = base64.b64encode(excel_buffer.getvalue()).decode()
                st.markdown("""
                    <div style="text-align: center;">
                        <p style="margin-bottom: 0.5rem; color: #666;">Detailed Report</p>
                        <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{}" download="test_cases_analysis.xlsx" class="download-button">
                            📊 Download Excel
                        </a>
                    </div>
                """.format(b64_excel), unsafe_allow_html=True)
            
            with col3:
                json_str = export_to_json(df)
                b64_json = base64.b64encode(json_str.encode()).decode()
                st.markdown("""
                    <div style="text-align: center;">
                        <p style="margin-bottom: 0.5rem; color: #666;">API Format</p>
                        <a href="data:application/json;base64,{}" download="test_cases_analysis.json" class="download-button">
                            🔄 Download JSON
                        </a>
                    </div>
                """.format(b64_json), unsafe_allow_html=True)
            
            with col4:
                html_content = export_to_html(df)
                b64_html = base64.b64encode(html_content.encode()).decode()
                st.markdown("""
                    <div style="text-align: center;">
                        <p style="margin-bottom: 0.5rem; color: #666;">Web Report</p>
                        <a href="data:text/html;base64,{}" download="test_cases_analysis.html" class="download-button">
                            🌐 Download HTML
                        </a>
                    </div>
                """.format(b64_html), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Store in history
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = {
                'timestamp': current_time,
                'total_cases': len(df),
                'high_priority': int(sum(df['priority'])),
                'avg_risk_score': float(df['risk_score'].mean()),
                'data': df.to_dict('records')
            }
            st.session_state.history.append(history_entry)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please check your CSV file format and data types")

elif page == "Historical Trends":
    st.markdown('<h1 class="main-header">Historical Trends</h1>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No historical data available yet. Please analyze some test cases first.")
    else:
        # Create trend data
        trend_data = pd.DataFrame([
            {
                'timestamp': entry['timestamp'],
                'total_cases': entry['total_cases'],
                'high_priority': entry['high_priority'],
                'avg_risk_score': entry['avg_risk_score']
            }
            for entry in st.session_state.history
        ])
        
        # Plot trends
        st.subheader("Trends Over Time")
        
        # Priority Distribution Trend
        fig = px.line(trend_data, x='timestamp', y=['total_cases', 'high_priority'],
                     title='Test Case Priority Distribution Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Average Risk Score Trend
        fig = px.line(trend_data, x='timestamp', y='avg_risk_score',
                     title='Average Risk Score Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed History
        st.subheader("Analysis History")
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Analysis {entry['timestamp']}"):
                st.write(f"Total Cases: {entry['total_cases']}")
                st.write(f"High Priority Cases: {entry['high_priority']}")
                st.write(f"Average Risk Score: {entry['avg_risk_score']:.2f}")

elif page == "Settings":
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    # Your existing JIRA settings are already in the sidebar

elif page == "Test Case Analysis":
    st.markdown('<h1 class="main-header">Test Case Analysis</h1>', unsafe_allow_html=True)
    
    # File Upload Section
    st.subheader("Upload Test Cases Data")
    
    # Add file format instructions
    st.markdown("""
    ### File Format Instructions
    Please upload a CSV file with the following columns:
    - `test_case_id`: Unique identifier for each test case (e.g., TC001)
    - `execution_result`: Test execution outcome (pass/fail)
    - `defect_severity`: Severity level of defects (low/medium/high)
    - `code_coverage`: Code coverage percentage (numeric value between 50-100)
    
    Download our sample template to get started:
    """)
    
    # Add sample template download button
    sample_path = "data/samples/sample_test_cases_template.csv"
    with open(sample_path, "r") as f:
        csv_content = f.read()
    b64_csv = base64.b64encode(csv_content.encode()).decode()
    st.markdown(f"""
        <a href="data:file/csv;base64,{b64_csv}" download="test_cases_template.csv" 
           style="display: inline-block; padding: 0.5rem 1rem; 
                  background-color: #4CAF50; color: white; 
                  text-decoration: none; border-radius: 4px;
                  margin-bottom: 1rem;">
            📥 Download Template CSV
        </a>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Process data
    df = None
    if uploaded_file is None and st.session_state.current_data is None:
        st.info("No file uploaded. Using sample test cases for demonstration.")
        df = pd.read_csv("data/sample_test_cases.csv")
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_data = df  # Store in session state
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        df = st.session_state.current_data
    
    if df is not None:
        try:
            required_columns = ['test_case_id', 'execution_result', 'defect_severity', 'code_coverage']
            
            # Validate required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            
            # Clean and standardize the data
            df['execution_result'] = df['execution_result'].str.lower()
            df['defect_severity'] = df['defect_severity'].str.lower()
            df['code_coverage'] = pd.to_numeric(df['code_coverage'], errors='coerce')
            
            # Fill any missing values
            df['code_coverage'] = df['code_coverage'].fillna(df['code_coverage'].mean())
            df['execution_result'] = df['execution_result'].fillna('pass')
            df['defect_severity'] = df['defect_severity'].fillna('low')
            
            # Calculate risk scores
            df['risk_score'] = df.apply(calculate_risk_score, axis=1)
            
            # Make predictions
            df_processed = df.copy()
            df_processed['defect_severity'] = df_processed['defect_severity'].map({'low': 1, 'medium': 2, 'high': 3})
            df_processed['execution_result'] = df_processed['execution_result'].map({'pass': 0, 'fail': 1})
            
            features = ['defect_severity', 'code_coverage', 'execution_result']
            predictions = st.session_state.model.predict(df_processed[features])
            df['priority'] = predictions
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Coverage Analysis", "Failure Patterns", "Risk Assessment"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Test Cases", len(df))
                with col2:
                    st.metric("High Priority Cases", sum(predictions))
                with col3:
                    st.metric("Average Risk Score", f"{df['risk_score'].mean():.2f}")
                
                # Priority Distribution
                fig = px.pie(df, names='priority', title='Priority Distribution',
                            color_discrete_map={0: '#3498db', 1: '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                st.markdown(get_csv_download_link(df), unsafe_allow_html=True)
                
                # JIRA Integration
                if st.button("Create JIRA Issues for High Priority Cases"):
                    jira = connect_to_jira()
                    if jira:
                        high_priority_cases = df[df['priority'] == 1]
                        for _, case in high_priority_cases.iterrows():
                            if create_jira_issue(jira, case['test_case_id'], case['risk_score']):
                                st.success(f"Created JIRA issue for test case {case['test_case_id']}")
            
            with tab2:
                # Coverage Analysis
                st.subheader("Code Coverage Analysis")
                
                # Coverage distribution
                fig = px.histogram(df, x='code_coverage', nbins=20,
                                 title='Code Coverage Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Coverage by severity
                fig = px.box(df, x='defect_severity', y='code_coverage',
                            title='Code Coverage by Defect Severity')
                st.plotly_chart(fig, use_container_width=True)
                
                # Coverage vs Risk Score
                fig = px.scatter(df, x='code_coverage', y='risk_score',
                               color='priority', title='Code Coverage vs Risk Score')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Failure Pattern Analysis
                st.subheader("Failure Pattern Analysis")
                
                patterns = detect_failure_patterns(df)
                for pattern in patterns:
                    st.info(pattern)
                
                # Failure trend
                fig = px.line(df, x=df.index, y='execution_numeric',
                             title='Test Execution Result Trend')
                st.plotly_chart(fig, use_container_width=True)
                
                # Failure correlation matrix
                correlation = df[['code_coverage', 'execution_numeric', 'priority', 'risk_score']].corr()
                fig = px.imshow(correlation, title='Correlation Matrix')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Risk Assessment
                st.subheader("Risk Assessment")
                
                # Risk score distribution
                fig = px.histogram(df, x='risk_score', nbins=20,
                                 title='Risk Score Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors breakdown by severity
                risk_by_severity = df.groupby('defect_severity')['risk_score'].mean().reset_index()
                fig = px.bar(risk_by_severity, x='defect_severity', y='risk_score',
                            title='Average Risk Score by Severity')
                st.plotly_chart(fig, use_container_width=True)
                
                # High risk test cases
                st.subheader("High Risk Test Cases")
                high_risk_cases = df[df['risk_score'] > df['risk_score'].quantile(0.75)]
                st.dataframe(high_risk_cases)
            
            # Add export options
            st.markdown("### Export Results")
            st.markdown('<div style="padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin: 1rem 0;">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                csv = df.to_csv(index=False)
                b64_csv = base64.b64encode(csv.encode()).decode()
                st.markdown("""
                    <div style="text-align: center;">
                        <p style="margin-bottom: 0.5rem; color: #666;">Basic Data</p>
                        <a href="data:file/csv;base64,{}" download="test_cases_analysis.csv" class="download-button">
                            📥 Download CSV
                        </a>
                    </div>
                """.format(b64_csv), unsafe_allow_html=True)
            
            with col2:
                excel_buffer = export_to_excel(df)
                b64_excel = base64.b64encode(excel_buffer.getvalue()).decode()
                st.markdown("""
                    <div style="text-align: center;">
                        <p style="margin-bottom: 0.5rem; color: #666;">Detailed Report</p>
                        <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{}" download="test_cases_analysis.xlsx" class="download-button">
                            📊 Download Excel
                        </a>
                    </div>
                """.format(b64_excel), unsafe_allow_html=True)
            
            with col3:
                json_str = export_to_json(df)
                b64_json = base64.b64encode(json_str.encode()).decode()
                st.markdown("""
                    <div style="text-align: center;">
                        <p style="margin-bottom: 0.5rem; color: #666;">API Format</p>
                        <a href="data:application/json;base64,{}" download="test_cases_analysis.json" class="download-button">
                            🔄 Download JSON
                        </a>
                    </div>
                """.format(b64_json), unsafe_allow_html=True)
            
            with col4:
                html_content = export_to_html(df)
                b64_html = base64.b64encode(html_content.encode()).decode()
                st.markdown("""
                    <div style="text-align: center;">
                        <p style="margin-bottom: 0.5rem; color: #666;">Web Report</p>
                        <a href="data:text/html;base64,{}" download="test_cases_analysis.html" class="download-button">
                            🌐 Download HTML
                        </a>
                    </div>
                """.format(b64_html), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Store in history
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = {
                'timestamp': current_time,
                'total_cases': len(df),
                'high_priority': int(sum(df['priority'])),
                'avg_risk_score': float(df['risk_score'].mean()),
                'data': df.to_dict('records')
            }
            st.session_state.history.append(history_entry)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please check your CSV file format and data types")

elif page == "Settings":
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    # Your existing JIRA settings are already in the sidebar
