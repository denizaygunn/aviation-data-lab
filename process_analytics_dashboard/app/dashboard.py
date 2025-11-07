import streamlit as st
import pandas as pd
import plotly.express as px

# Page title
st.title("Aviation Process Analytics Dashboard")

# --- About Section ---
st.markdown("""
### About this Dashboard
This interactive Streamlit dashboard visualizes production data from an aviation manufacturing process and uses a **Linear Regression model** to predict defect rates based on assembly time and production volume.

It includes:
-  Data visualization of machine performance  
-  Machine learning–based defect rate prediction  
-  Model performance metrics and comparison between actual and predicted values  

This project demonstrates how data-driven insights and predictive analytics can improve aviation manufacturing efficiency.
""")
# --- Tech Stack Section ---
st.markdown("""
### Tech Stack
- **Python** – Data processing and machine learning  
- **Pandas** – Data manipulation  
- **Plotly** – Interactive visualizations  
- **Streamlit** – Web-based dashboard interface  
- **Scikit-learn** – Linear Regression model and evaluation metrics  
""")
# Load example production data
df = pd.read_csv("process_analytics_dashboard/data/production_data.csv")

st.header("Production Overview")
st.write(df.head())

# Plot: Defect rate vs Assembly time
fig = px.scatter(
    df,
    x="assembly_time",
    y="defect_rate",
    color="machine_id",
    size="production_volume",
    title="Defect Rate vs Assembly Time"
)
st.plotly_chart(fig)

# Summary metrics
avg_defect = round(df["defect_rate"].mean(), 3)
avg_time = round(df["assembly_time"].mean(), 1)

st.metric("Average Assembly Time", f"{avg_time} min")
st.metric("Average Defect Rate", f"{avg_defect}")

# --- Machine Learning Prediction Section ---
st.subheader("Defect Rate Prediction (Machine Learning)")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Prepare training data
X = df[["assembly_time", "production_volume"]]
y = df["defect_rate"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write(f"**Model Performance:** R² = {r2:.3f}, MSE = {mse:.4f}")

# Interactive prediction input
st.write("### Enter new production data to predict the defect rate:")
time_input = st.number_input("Assembly Time (minutes):", min_value=10, max_value=100, value=40)
volume_input = st.number_input("Production Volume:", min_value=10, max_value=500, value=100)

# Make a prediction for the given inputs
prediction = model.predict([[time_input, volume_input]])[0]

st.success(f"Predicted Defect Rate: {prediction:.3f}")

# --- Model Performance Visualization ---
st.write("### Actual vs Predicted Defect Rate")
# --- Model Performance Metrics ---
st.write("### Model Performance Summary")
col1, col2 = st.columns(2)
col1.metric(label="R² Score", value=f"{r2:.3f}")
col2.metric(label="Mean Squared Error", value=f"{mse:.5f}")

import matplotlib.pyplot as plt

# Create a scatter plot: actual vs predicted
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predictions")
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Perfect Prediction"
)

ax.set_xlabel("Actual Defect Rate")
ax.set_ylabel("Predicted Defect Rate")
ax.legend()
st.pyplot(fig)
# --- Data & Model Information ---
st.markdown("---")
st.markdown("""
### Data & Model Information
- **Dataset:** Simulated aviation production data containing machine IDs, assembly times, defect rates, and production volumes.  
- **Model Used:** *Linear Regression* – predicts defect rate based on assembly time and production volume.  
- **Evaluation Metrics:** R² Score and Mean Squared Error (MSE).  
- **Purpose:** Demonstrate how data-driven insights can improve aviation manufacturing quality and process optimization.  
""")