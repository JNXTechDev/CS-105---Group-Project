import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title("SUPERMARKET SALES: MARKET SEGMENTATION ANALYSIS")

# Step 1: Upload the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Step 2: Data Cleaning
    st.subheader("Data Cleaning")
    if st.checkbox("Check for missing values"):
        missing_count = df.isnull().sum()
        st.write(missing_count)

    # Step 3: Encode Categorical Variables
    st.subheader("Encoding Categorical Variables")
    categorical_columns = st.multiselect(
        "Select categorical columns to encode:",
        options=df.select_dtypes(include=["object"]).columns.tolist(),
    )
    if categorical_columns:
        df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)
        st.write("Encoded Data Preview:")
        st.write(df_encoded.head())
    else:
        df_encoded = df

    # Step 4: Normalize Numerical Data
    st.subheader("Normalizing Data")
    numerical_columns = st.multiselect(
        "Select numerical columns to normalize:",
        options=df_encoded.select_dtypes(include=[np.number]).columns.tolist(),
    )
    if numerical_columns:
        scaler = MinMaxScaler()
        df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])
        st.write("Normalized Data Preview:")
        st.write(df_encoded.head())

    # Step 5: K-Means Clustering
st.subheader("K-Means Clustering")
if st.checkbox("Run K-Means Clustering"):
    # Drop non-numeric columns
    non_numeric_cols = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        df_encoded = df_encoded.drop(non_numeric_cols, axis=1)
        st.write(f"Dropped non-numeric columns: {non_numeric_cols}")

    # Check for missing values
    if df_encoded.isnull().any().any():
        st.warning("Dataset contains missing values. Dropping rows with missing values.")
        df_encoded = df_encoded.dropna()

    # Ensure all columns are numeric
    if not all([np.issubdtype(dtype, np.number) for dtype in df_encoded.dtypes]):
        st.error("Dataset still contains non-numeric data. Please preprocess your data correctly.")
    else:
        # Select number of clusters
        n_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=4)

        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_encoded["Cluster"] = kmeans.fit_predict(df_encoded)

        # Display cluster assignments
        st.write("Clustered Data:")
        st.write(df_encoded.head())

        # Visualize clusters with PCA
        st.subheader("Cluster Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df_encoded.drop(columns=["Cluster"]))
        df_encoded["PCA1"], df_encoded["PCA2"] = X_pca[:, 0], X_pca[:, 1]

        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=df_encoded, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", ax=ax, s=100, alpha=0.7)
        plt.title("Customer Segments Visualization using PCA")
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        st.pyplot(fig)