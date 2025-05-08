import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

st.title('Patient Death Prediction using SVM')

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("dataset.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_clean = df[numeric_cols].drop(columns=["hospital_id", "icu_id", "patient_id"], errors='ignore').dropna()
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
    
    x = df_scaled.iloc[:, :-1]
    y = df_scaled.iloc[:, -1]
    y_encoded = LabelEncoder().fit_transform(y)
    
    return x, y_encoded, df_scaled

@st.cache_resource
def train_svm_model(x_train, y_train):
    model = SVC(random_state=42)
    model.fit(x_train, y_train)
    return model

# Load and preprocess data
x, y_encoded, df_scaled = load_and_preprocess_data()

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.1, random_state=42)

# Train SVM model
with st.spinner("SVM model training..."):
    svm_model = train_svm_model(x_train, y_train)
st.success("SVM model trained")

# Evaluate
y_pred = svm_model.predict(x_test)
st.text(f"Accuracy: {accuracy_score(y_test, y_pred)}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.text("Confusion Matrix:")
st.text(confusion_matrix(y_test, y_pred))

# User input for predictions
edited_df = pd.DataFrame(columns=df_scaled.columns[:-1])
x_user = st.data_editor(edited_df, num_rows="dynamic")

if not x_user.empty and x_user.dropna().shape[0] > 0:
    y_pred_user = svm_model.predict(x_user)
    st.write("Predicted class is", y_pred_user[0])
else:
    st.write("Please input data to make a prediction.")
