# Aviation Process Analytics Dashboard

An interactive Streamlit dashboard that visualizes aviation production data and predicts defect rates using a **Linear Regression model**.

## Goal
To demonstrate how data-driven insights and machine learning can improve efficiency and quality in aviation manufacturing processes.

## Features
- Visualization of production performance (assembly time, defect rate, production volume)  
- Predictive model for defect rate estimation  
- Model evaluation with R² and MSE metrics  
- Interactive Streamlit interface  

## Tech Stack
- **Python** – Data processing and machine learning  
- **Pandas** – Data manipulation  
- **Plotly** – Interactive visualizations  
- **Streamlit** – Web-based dashboard interface  
- **Scikit-learn** – Linear Regression model and evaluation metrics  

## How to Run
```bash
cd process_analytics_dashboard
pip install -r requirements.txt
streamlit run app/dashboard.py