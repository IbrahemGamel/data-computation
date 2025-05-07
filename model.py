import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


st.title('Patient Death Perdiction using SVM')

# cleaning dataframe
df = pd.read_csv("dataset.csv")
important_df = pd.read_csv("important_features.csv")

# Reducing features to 20 so it's easier for user to input them
df_clean = df[df.columns.intersection(important_df["Feature"])]


df_clean.dropna(inplace=True)
df_clean = df_clean.iloc[:10000, :]
# Dimensionality reduction
with st.spinner("LDA reduction...", show_time=True):
    x = df_clean.iloc[:, :-1]
    y = df_clean.iloc[:, -1]   
    encoded = LabelEncoder()
    y_encoded = encoded.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

    lda = LDA(n_components=1)
    x_train_lda = lda.fit_transform(x_train, y_train)

st.success("LDA reduction done")

# SVM model

with st.spinner("SVM model training...", show_time=True):
    svm_model = SVC(random_state=42)
    svm_model.fit(x_train_lda, y_train)
st.success("SVM model trained")

with st.spinner("Using GridSearch to find best model parameters...", show_time=True):
    param_grid = {
        'C': [0.1, 1],  
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale'], 
    }
    ## we use GridSearchCV to find the best model parameters for optimization

    grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', verbose =1, n_jobs = -1)
    grid_search.fit(x_train_lda, y_train)
    best_svm_model = grid_search.best_estimator_
st.success("Found and applied best model params")

variable_dict = {}
for i in df_clean.iloc[:, :-1].columns:
    variable_dict[i] = 0

edited_df = pd.DataFrame.from_dict(variable_dict, orient="index").T
x_test = st.data_editor(edited_df)

x_test_lda = lda.transform(x_test)
st.write("LDA value is", x_test_lda[0][0])
y_pred = best_svm_model.predict(x_test_lda)
st.write("Predicted class is", y_pred[0])
