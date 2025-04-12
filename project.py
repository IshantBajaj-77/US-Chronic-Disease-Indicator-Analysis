import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  # For interactive dashboard
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# Load dataset
df = pd.read_csv("U.S._Chronic_Disease_Indicators.csv")

# Basic Info
print("\n shape of data")
print(df.shape)

# Display column names
print("\n Columns in Dataset:")
print(df.columns)

# Display dataset info
print("\nData types & non-null counts")
print(df.info())

# Display summary stats
print("\n Stats summary (mean, std, min, max")
print(df.describe())

# Display first few rows
print("\n First 5 rows of dataset:")
print(df.head())

# Check Missing Values
print("\n Check Missing Values",df.isnull().sum())

#Drop unnecessary columns
columns_to_drop = [
    "Response", "ResponseID", "StratificationCategory2", "Stratification2",
    "StratificationCategory3", "Stratification3", "StratificationCategoryID2",
    "StratificationID2", "StratificationCategoryID3", "StratificationID3",
    "DataValueFootnote", "DataValueFootnoteSymbol", "Geolocation"
]
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

# Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

# Print new shape & remaining columns
print("\n New DataFrame Shape:", df_cleaned.shape)
print(" Remaining Columns:", df_cleaned.columns)

# Display dataset info after cleaning
print("\n Dataset Info After Cleaning:")
df_cleaned.info()

# Summary statistics after cleaning
print("\n Summary Statistics After Cleaning:")
print(df_cleaned.describe())

# Check missing values after cleaning
print("\n Missing Values After Cleaning:")
print(df_cleaned.isnull().sum())

# Drop rows with missing values
df_cleaned.dropna(inplace=True)

# Drop columns where more than 50% values are missing
df_cleaned = df_cleaned.dropna(axis=1, thresh=int(0.5 * len(df_cleaned)))

# Check missing values after cleaning
print("\n Missing Values After Cleaning:")
print(df_cleaned.isnull().sum())

# Convert categorical variables into numerical values
categorical_cols = df_cleaned.select_dtypes(include='object').columns
print("\n Categorical Columns:", categorical_cols)
df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

# ------------------ Inferential Statistics ------------------
print("\nInferential Statistics:")
num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
if len(num_cols) >= 2:
    group1 = df_cleaned[num_cols[0]].sample(30, random_state=1)
    group2 = df_cleaned[num_cols[1]].sample(30, random_state=1)

    # T-test
    t_stat, t_p = stats.ttest_ind(group1, group2)
    print(f"T-test between {num_cols[0]} and {num_cols[1]}: t-statistic = {t_stat:.4f}, p-value = {t_p:.4f}")

    # Z-test
    z_stat = (group1.mean() - group2.mean()) / np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
    z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"Z-test: z-statistic = {z_stat:.4f}, p-value = {z_p:.4f}")

# Chi-squared test for categorical variables
if "TopicType" in df.columns and "DataSource" in df.columns:
    contingency_table = pd.crosstab(df["TopicType"], df["DataSource"])
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency_table)
    print(f"Chi-squared test between 'TopicType' and 'DataSource': chi2 = {chi2:.4f}, p-value = {chi_p:.4f}")

# VIF Calculation (Multicollinearity Check)
print("\nVariance Inflation Factor (VIF):")
X = df_cleaned[num_cols].dropna()
X = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# Seaborn-based Histograms for Numerical Features
plt.figure(figsize=(15, 10), dpi=100)
sampled_df = df_cleaned.sample(n=5000, random_state=42)
num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns[:5]

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(sampled_df[col], bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.suptitle("Seaborn Histograms of Numerical Features", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#  Improved Line Plot for Disease Trends
plt.figure(figsize=(16, 8)) 

sns.lineplot(
    x="YearStart", 
    y="DataValue", 
    hue="Topic", 
    data=df, 
    palette="coolwarm", 
    linewidth=2
)

plt.title("Chronic Disease Trend Over Time", fontsize=16, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Data Value")
plt.xticks(rotation=45)

plt.legend(
    title="Disease Type", 
    bbox_to_anchor=(1.25, 1), 
    loc='upper left',
    borderaxespad=0.5,
    fontsize='small',  
    title_fontsize='medium'
)

plt.tight_layout()  # Auto adjust for spacing
plt.show()

# Seaborn Barplot - State-wise Diabetes Cases
plt.figure(figsize=(14, 7))
top_states = (
    df[df["Topic"] == "Diabetes"]
    .groupby("LocationDesc")["DataValue"]
    .mean()
    .nlargest(10)
).reset_index()

sns.barplot(x="LocationDesc", y="DataValue", hue="LocationDesc",data=top_states, palette="Reds_r", edgecolor='black', legend=False)
plt.title("Top 10 States with Highest Average Diabetes Cases", fontsize=16, fontweight='bold')
plt.xlabel("State", fontsize=12)
plt.ylabel("Average Data Value", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Distribution of Chronic Disease Records by Year
plt.figure(figsize=(12, 6))
sns.countplot(
    x="YearStart",hue="YearStart",
    data=df,
    palette="viridis",
    order=sorted(df["YearStart"].unique()),
    legend=False
)
plt.title("Distribution of Chronic Disease Records by Year", fontsize=16, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#  Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_cleaned.corr(), annot=False, cmap="coolwarm", linewidths=0.5, cbar=True)
plt.title(" Correlation Heatmap", fontsize=16, fontweight='bold')
plt.show()

# Improved Boxplot for Outlier Detection
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols[:6], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df_cleaned[col], color='cyan')
    plt.title(f"Outlier Detection - {col}")
plt.tight_layout()
plt.show()

# Interactive Dashboard using Streamlit
def main():
    st.title(" Chronic Disease Analysis Dashboard")

    if st.checkbox("Show Raw Data"):
        st.write(df_cleaned.head())

    st.subheader("Disease Trends Over Time")
    disease_options = df["Topic"].unique()
    selected_disease = st.selectbox("Select a Disease", disease_options)
    filtered_df = df[df["Topic"] == selected_disease]
    st.line_chart(filtered_df.set_index("YearStart")["DataValue"])

    st.subheader("State-wise Analysis")
    top_states = df[df["Topic"] == selected_disease].groupby("LocationDesc")["DataValue"].mean().nlargest(10)
    st.bar_chart(top_states)

if __name__ == "__main__":
    main()

print("\nEDA Completed! Data is now cleaned and visualized successfully.")
