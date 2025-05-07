from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
import io
import streamlit as st

df = pd.read_csv("dataset.csv")
df.info()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df_clean = df.drop(columns=["hospital_id","icu_id", "patient_id"]) # very very large variance and doesn't effect the predection so it will be dropped for easier diagrams
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
df_clean = df[numeric_cols].dropna()


x = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]   
encoded = LabelEncoder()
y_encoded = encoded.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

lda = LDA(n_components=1)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)


st.code(
    '''
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


svm_model = SVC(random_state=42)
    '''
)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



st.subheader('model training')
st.code(
    '''
    # training our SVM model using the LDA-transformed features
svm_model.fit(x_train_lda, y_train)
'''
)

# training our SVM model using the LDA-transformed features
with st.spinner("SVM model training...", show_time=True):
    svm_model = SVC(random_state=42)
    svm_model.fit(x_train_lda, y_train)
st.success("SVM model trained")

st.subheader('Hyperparameter Tuning')

st.code(
    '''
    # we use GridSearchCV to find the best model parameters for optimization
param_grid = {
    'C': [0.1, 1],  
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale'], 
}

grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', verbose =1, n_jobs = -1)

grid_search.fit(x_train_lda, y_train)


print(f"Best parameters: {grid_search.best_params_}")

best_svm_model = grid_search.best_estimator_
'''
)

# we use GridSearchCV to find the best model parameters for optimization
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

st.markdown(f"Best parameters: :green[{grid_search.best_params_}]")

st.subheader('Model evaluation')
st.code(
    '''
    y_pred = best_svm_model.predict(x_test_lda)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}") #accuracy
print("\nClassification Report:")
print(classification_report(y_test, y_pred)) # precision, recall, f1-score
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
    '''
)

y_pred = best_svm_model.predict(x_test_lda)

st.markdown(f"Accuracy: :green[{accuracy_score(y_test, y_pred)}]") #accuracy
st.markdown("Classification Report:")
st.code(classification_report(y_test, y_pred)) # precision, recall, f1-score
st.markdown(f"Confusion Matrix: ")
st.code(confusion_matrix(y_test, y_pred)) # precision, recall, f1-score


st.subheader('Visualizing results')
st.code(
    '''
    plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Survived', 'Died'], yticklabels=['Survived', 'Died'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
'''
)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Survived', 'Died'], yticklabels=['Survived', 'Died'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt)