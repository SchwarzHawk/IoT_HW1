import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CRISP-DM: Deployment
# This Streamlit app serves as the deployment of the data product.
st.title("Interactive Linear Regression Visualizer")

# CRISP-DM: Business Understanding
# The goal is to create an interactive tool to understand linear regression,
# the effect of its parameters, and how outliers are identified.

# --- User Interface (Streamlit) ---
# User input for parameters
st.sidebar.header("Model Parameters")
n = st.sidebar.slider("Number of data points (n)", min_value=100, max_value=1000, value=200)
a = st.sidebar.slider("True Coefficient 'a'", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
b_true = 5  # Fixed intercept for simplicity
variance = st.sidebar.slider("Noise Variance (var)", min_value=0, max_value=1000, value=100)

# --- CRISP-DM: Data Understanding & Preparation ---
# Generate synthetic data based on user-defined parameters.
# This combines understanding the data's structure (y = ax + b + noise)
# and preparing it for the model.
@st.cache_data
def generate_data(n, a, b, variance_val):
    np.random.seed(42)  # for reproducibility
    x = np.random.rand(n) * 10
    noise = np.random.normal(0, np.sqrt(variance_val), n)
    y = a * x + b + noise
    return x, y

x, y = generate_data(n, a, b_true, variance)

# Reshape x for sklearn
x_reshaped = x.reshape(-1, 1)

# --- CRISP-DM: Modeling ---
# Perform linear regression
model = LinearRegression()
model.fit(x_reshaped, y)
y_pred = model.predict(x_reshaped)

# --- CRISP-DM: Evaluation ---
# Evaluate the model by identifying points with the largest error (residuals).
# Calculate residuals
residuals = np.abs(y - y_pred)

# Identify top 5 outliers
outlier_indices = np.argsort(residuals)[-5:]

# --- Visualization ---
st.header("Linear Regression Plot")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, alpha=0.6, label="Generated Data Points")
ax.plot(x_reshaped, y_pred, color='red', linewidth=2, label="Linear Regression Line")

# Highlight outliers
ax.scatter(x[outlier_indices], y[outlier_indices], color='purple', s=100, marker='o', edgecolor='black', label="Top 5 Outliers")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Linear Regression and Outlier Identification")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Displaying Results ---
st.header("Regression Results")
st.write(f"**True Equation:** Y = {a:.2f}X + {b_true}")
st.write(f"**Estimated Regression Equation:** Y = {model.coef_[0]:.2f}X + {model.intercept_:.2f}")
st.write(f"**Coefficient 'a' :** {model.coef_[0]:.2f}")
st.write(f"**Intercept 'b' :** {model.intercept_:.2f}")

# --- Displaying Outlier Information ---
st.header("Top 5 Outliers")
outlier_data = {
    'X Value': x[outlier_indices],
    'Y Value': y[outlier_indices],
    'Residual (Distance)': residuals[outlier_indices]
}
outlier_df = pd.DataFrame(outlier_data).sort_values(by='Residual (Distance)', ascending=False).reset_index(drop=True)
st.dataframe(outlier_df.style.format("{:.2f}"))
