# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App title and description
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.write("""
This app allows you to:
1. Upload and visualize customer data.
2. Train a model to predict customer churn risk.
3. Use the model to make predictions interactively.
""")

# Sidebar for user inputs
st.sidebar.header("User Inputs")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# App logic
if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")

    # Sidebar options
    option = st.sidebar.radio("Choose a feature:", ["Dataset Overview", "Visualizations", "Prediction"])

    if option == "Dataset Overview":
        # Display dataset preview
        st.subheader("Dataset Overview")
        st.dataframe(df)
        st.write("### Dataset Summary")
        st.write(df.describe())

    elif option == "Visualizations":
        # Visualizations
        st.subheader("Visualizations")

        # Choose columns to visualize
        column = st.selectbox("Select a column to visualize:", df.columns)

        # Generate plots
        if df[column].dtype in ["int64", "float64"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[column], kde=True, ax=ax, color="skyblue")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=column, data=df, palette="viridis", ax=ax)
            st.pyplot(fig)

    elif option == "Prediction":
        # Data preprocessing
        st.subheader("Prediction")
        st.write("### Preprocessing and Training the Model")

        if "churn_risk_score" not in df.columns:
            st.error("The dataset must include a 'churn_risk_score' column for predictions.")
        else:
            # Drop irrelevant columns
            unused_columns = ['security_no', 'referral_id', 'last_visit_time']
            df = df.drop(columns=unused_columns, errors='ignore')
            
            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))

            # Convert categorical variables to numerical
            df = pd.get_dummies(df, drop_first=True)

            # Splitting data into features and labels
            X = df.drop(columns=['churn_risk_score'])
            y = df['churn_risk_score']

            # Split into training and testing data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2%}")

            # Interactive prediction form
            st.write("### Make a Prediction")
            input_data = {}
            for column in X.columns:
                input_data[column] = st.sidebar.number_input(f"Enter {column}", value=0.0)

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                st.success(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")

else:
    st.info("Please upload a dataset to proceed.")
