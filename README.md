# Interactive Linear Regression Visualizer

This project is an interactive web application built with Streamlit that allows users to visualize the concepts of linear regression. Users can dynamically generate data, adjust model parameters, and see how changes in slope, noise, and data size affect the regression line and outlier identification.

The project structure and code comments follow the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology.

## Features

-   **Interactive Data Generation**:
    -   Select the number of data points (`n`) from 100 to 1000.
    -   Adjust the true coefficient (`a`) of the linear relationship.
    -   Control the amount of random noise (`variance`) added to the data.
-   **Real-time Visualization**:
    -   Scatter plot of the generated data points.
    -   Calculated linear regression line drawn over the data.
-   **Outlier Detection**:
    -   Automatically identifies and highlights the top 5 outliers (points furthest from the regression line).
    -   Displays a detailed table with the coordinates and residual values for each outlier.
-   **Model Insights**:
    -   Shows the "true" underlying equation used for data generation.
    -   Displays the estimated regression equation calculated by the model for comparison.

## How to Run

1.  **Prerequisites**:
    -   Python 3.7+
    -   pip

2.  **Installation**:
    Install the required Python packages using pip:
    ```bash
    pip install streamlit numpy pandas scikit-learn matplotlib
    ```

3.  **Execution**:
    Navigate to the project directory in your terminal and run the following command:
    ```bash
    streamlit run IoT_HW1.py
    ```
    The application will automatically open in a new tab in your web browser.

## CRISP-DM in this Project

-   **Business Understanding**: The goal is to create an educational tool to explore linear regression concepts interactively.
-   **Data Understanding & Preparation**: Synthetic data `y = ax + b + noise` is generated and prepared based on user-defined parameters.
-   **Modeling**: A `LinearRegression` model from `scikit-learn` is trained on the generated data.
-   **Evaluation**: The model is evaluated visually, and outliers are identified by calculating the largest residuals.
-   **Deployment**: The Streamlit application itself serves as the deployment of the final data product, making it accessible to end-users.
