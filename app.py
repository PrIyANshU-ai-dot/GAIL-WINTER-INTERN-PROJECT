import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.optimize import minimize


model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("polynomial_features.pkl")


features = [
    "production_value", "product_emissions_MtCO2", "flaring_emissions_MtCO2",
    "venting_emissions_MtCO2", "own_fuel_use_emissions_MtCO2", "fugitive_methane_emissions_MtCO2e",
    "temperature", "process_efficiency", "equipment_age", "renewable_energy_share"
]


st.title("Emissions Prediction & Optimization]")


df = pd.read_csv('data/emissions_high_granularity.csv')


commodity_emissions = df.groupby('commodity').agg(
    total_emissions=('total_emissions_MtCO2e', 'sum'),
    total_production=('production_value', 'sum')
).reset_index()


st.write("### Total Emissions Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.histplot(df['total_emissions_MtCO2e'], kde=True, bins=30, ax=ax1)
ax1.set_title("Total Emissions Distribution")
st.pyplot(fig1)
st.write("""
This histogram shows the distribution of total emissions across the dataset. The **kde curve** helps visualize the density 
of emissions and their frequency, giving a clear idea of where most of the emissions lie.
""")


st.write("### Total Emissions by Commodity")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x='commodity', y='total_emissions', data=commodity_emissions, ax=ax2)
ax2.set_title("Total Emissions by Commodity")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig2)
st.write("""
This bar chart shows the total emissions by commodity. It helps identify which commodities are contributing the most to total emissions.
Commodities with higher bars have a greater environmental impact.
""")


st.write("### Total Emissions Over Time")
fig3, ax3 = plt.subplots(figsize=(10, 6))
time_series_emissions = df.groupby('year')['total_emissions_MtCO2e'].sum().reset_index()
sns.lineplot(x='year', y='total_emissions_MtCO2e', data=time_series_emissions, ax=ax3)
ax3.set_title("Total Emissions Over Time")
st.pyplot(fig3)
st.write("""
This line plot shows the trend of total emissions over time. It helps to understand how emissions have changed in the past years 
and whether they are increasing or decreasing.
""")


st.write("### Enhanced Feature Correlation Heatmap")
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr()

heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    square=True, 
    cbar_kws={"shrink": 0.8},
    linewidths=0.5, 
    linecolor="white", 
    ax=heatmap_ax
)
heatmap_ax.set_title("Feature Correlation Heatmap", fontsize=14)
heatmap_ax.tick_params(axis="x", labelrotation=45)
heatmap_ax.tick_params(axis="y", labelrotation=0)

st.pyplot(heatmap_fig)
st.write("""
This heatmap provides insights into the relationships between numerical features in the dataset. 
Features with strong correlations (positive or negative) may indicate dependencies that the model can exploit.
""")


st.write("### Input Feature Values")
bar_fig, bar_ax = plt.subplots()
bar_ax.barh(features, np.random.rand(len(features)), color='skyblue')  # Placeholder for feature values
bar_ax.set_xlabel("Feature Value")
bar_ax.set_ylabel("Feature Name")
st.pyplot(bar_fig)
st.write("""
This bar chart provides a simple view of the input feature values. 
It allows you to visually compare the relative magnitude of each input feature.
""")


st.write("### Feature Trend Radar Chart")
def radar_chart(features, values):
    # Number of variables
    num_vars = len(features)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The values are circular, so we need to append the first value to the end
    values = np.concatenate((values, [values[0]]))
    angles += angles[:1]  # Close the circle

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)  # Line for features
    ax.set_yticklabels([])  # No radial axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.set_title("Feature Trend Radar Chart", fontsize=14)

    return fig


feature_values = np.random.rand(len(features))


radar_fig = radar_chart(features, feature_values)

st.pyplot(radar_fig)
st.write("""
This radar chart provides a visual comparison of the feature values. Each axis represents a feature, 
and the values are plotted in a circular manner, helping to understand how the features vary in relation to each other.
""")


st.header("Optimization for Emissions Reduction")
st.write("""
Specify the percentage reduction in emissions you would like to achieve. The app will then suggest optimized feature values to meet that target.
""")


target_reduction = st.slider("Target Emissions Reduction (%)", 0, 50, 10)


def objective(x):
    # Predict emissions from the model based on the input 'x' (feature values)
    input_scaled = scaler.transform(poly.transform([x]))
    return model.predict(input_scaled)[0]


constraints = (
    {"type": "ineq", "fun": lambda x: x[0] - 0.2},  # production_value >= 0.2
    {"type": "ineq", "fun": lambda x: 0.8 - x[9]},  # renewable_energy_share <= 0.8
)


initial_guess = [1.0] * len(features)
result = minimize(objective, initial_guess, constraints=constraints)


if result.success:
    optimized_values = result.x
    optimized_emissions = objective(optimized_values)
    st.write(f"### Optimized Emissions: {optimized_emissions:.2f} MtCO2e")
    

    optimized_df = pd.DataFrame([optimized_values], columns=features)
    st.write("### Optimized Feature Values")
    st.write(optimized_df)

st.header("Input Features")
st.write("## Instructions")
st.write("""
Enter feature values for all the input fields, then click on the **'Predict Emissions'** button. 
The app will show you the predicted emissions along with various plots to help you understand the prediction results, 
including a waterfall plot of feature importance, a heatmap of feature correlations, bar charts, and radar charts.
""")
st.write("""
Fill in the values for each of the features below. These inputs will help predict the emissions from natural gas processing operations. 
Make sure to enter the correct units as indicated next to each input field.
""")

user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", value=1.0, format="%.2f")


input_df = pd.DataFrame([user_input])
input_poly = poly.transform(input_df)
input_scaled = scaler.transform(input_poly)

if st.button("Predict Emissions"):
    
    prediction = model.predict(input_scaled)[0]
    st.write(f"### Predicted Emissions: {prediction:.2f} MtCO2e")


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    st.write("### Feature Importance (SHAP Values) - Waterfall Plot")
    shap_fig, shap_ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, feature_names=poly.get_feature_names_out(features)),
        max_display=10
    )
    st.pyplot(shap_fig)
    st.write("""
    The waterfall plot explains how each feature contributes to the predicted emissions for the current input.
    It shows the individual impact of each feature and how they push the prediction higher or lower.
    """)


    st.write("### Actual vs. Predicted Emissions")
    X_test = pd.DataFrame([[1.5, 2.1, 1.2, 0.3, 1.0, 0.5, 28.0, 85.0, 12, 30]])  # Example input
    y_test = np.array([2.5])  


    X_test_poly = poly.transform(X_test)
    X_test_scaled = scaler.transform(X_test_poly)


    y_pred = model.predict(X_test_scaled)

    comparison_df = pd.DataFrame({
        "Actual Emissions (MtCO2e)": y_test,
        "Predicted Emissions (MtCO2e)": y_pred
    })

    comparison_fig, comparison_ax = plt.subplots(figsize=(8, 6))

    
    comparison_df.plot(kind="bar", ax=comparison_ax, color=['skyblue', 'lightcoral'], width=0.7)
    comparison_ax.set_ylabel("Emissions (MtCO2e)", fontsize=12)
    comparison_ax.set_title("Actual vs. Predicted Emissions", fontsize=14)
    comparison_ax.set_xticklabels(['Emissions'], rotation=0)
    comparison_ax.legend(["Actual", "Predicted"], loc="upper left")

    
    st.pyplot(comparison_fig)

    st.write("""
    This bar chart compares the actual emissions with the predicted emissions from the model.
    - The **blue bar** represents the **actual emissions**.
    - The **red bar** represents the **predicted emissions**.
    The closer the bars are, the more accurate the prediction is.
    """)

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Results",
        data=convert_df_to_csv(comparison_df),
        file_name='prediction_results.csv',
        mime='text/csv',
    )
