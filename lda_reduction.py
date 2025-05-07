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
    x = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]   
encoded = LabelEncoder()
y_encoded = encoded.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

lda = LDA(n_components=1)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)
'''
)
st.markdown(f"Original feature shape: :green[{x_train.shape}]")
st.markdown(f"Reduced feature shape after LDA: :green[{x_train_lda.shape}]")

st.code(
    '''
    plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=x_train.iloc[:, 0], y=x_train.iloc[:, 1], hue=y_train, palette='Set1', alpha=0.6)
plt.title("Before LDA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Class", loc='best')

plt.subplot(1, 2, 2)
sns.scatterplot(x=x_train_lda[:, 0], y=[0]*len(x_train_lda), hue=y_train, palette='Set1', alpha=0.6)
plt.title("After LDA")
plt.xlabel("LDA Component")
plt.yticks([])  
plt.legend(title="Class", loc='best')

plt.tight_layout()
plt.show()
'''
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=x_train.iloc[:, 0], y=x_train.iloc[:, 1], hue=y_train, palette='Set1', alpha=0.6)
plt.title("Before LDA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Class", loc='best')

plt.subplot(1, 2, 2)
sns.scatterplot(x=x_train_lda[:, 0], y=[0]*len(x_train_lda), hue=y_train, palette='Set1', alpha=0.6)
plt.title("After LDA")
plt.xlabel("LDA Component")
plt.yticks([])  
plt.legend(title="Class", loc='best')

plt.tight_layout()

st.pyplot(plt)