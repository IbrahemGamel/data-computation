import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
import io
import streamlit as st
warnings.filterwarnings("ignore")


st.code(
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
import io
import streamlit as st
'''
)

st.subheader('2.1 General information')
st.code(
'''
df = pd.read_csv("dataset.csv")  
df.info()
'''
)
df = pd.read_csv("dataset.csv") # change 

buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text("DataFrame Info:")
st.code(info_str)

st.code(
'''
df.describe(include='all')
'''
)
st.dataframe(df.describe(include='all'))

st.code(
    '''
    # dropping the unamed col as it's all None
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    '''
)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

st.subheader("2.2 Target Variable Analysis ( hospital_death )")
st.code(
'''
target_counts = df['hospital_death'].value_counts()
target_percent = df['hospital_death'].value_counts(normalize=True) * 100
print(pd.concat([target_counts, target_percent], axis=1, keys=['Count', 'Percentage']))
'''
)
target_counts = df['hospital_death'].value_counts()
target_percent = df['hospital_death'].value_counts(normalize=True) * 100
st.dataframe(pd.concat([target_counts, target_percent], axis=1, keys=['Count', 'Percentage']))

st.code(
'''
sns.countplot(x='hospital_death', data=df)
plt.title('Distribution of Hospital Death') 
plt.show()
'''
)
sns.countplot(x='hospital_death', data=df)
plt.title('Distribution of Hospital Death') 
st.pyplot(plt)
plt.clf()  # Clear current figure

st.subheader("2.3 Missing Values")
st.code(
'''
missing_values = df.isnull().sum()
print("Missing Values per Column:")
print(missing_values[missing_values > 0])
'''
)

missing_values = df.isnull().sum()
st.text("Missing Values per Column:")
st.code(missing_values[missing_values > 0])

st.code(
'''
# Missing values percentage bar plot
missing_percent = df.isnull().mean().sort_values(ascending=False) * 100
missing_percent = missing_percent[missing_percent > 0]

plt.figure(figsize=(30, 40))
missing_percent.plot(kind='barh', fontsize=16)
plt.title('Percentage of Missing Values by Column', fontsize=28)
plt.xlabel('Percentage Missing (%)', fontsize=28)
plt.ylabel('Columns', fontsize=28)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
'''
)
# Missing values percentage bar plot
missing_percent = df.isnull().mean().sort_values(ascending=False) * 100
missing_percent = missing_percent[missing_percent > 0]
plt0 = plt.figure(figsize=(30, 40))
missing_percent.plot(kind='barh', fontsize=16)
plt.title('Percentage of Missing Values by Column', fontsize=28)
plt.xlabel('Percentage Missing (%)', fontsize=28)
plt.ylabel('Columns', fontsize=28)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
st.pyplot(plt0)
plt.clf()  # Clear current figure

st.code(
    '''
    df_clean = df.drop(columns=["hospital_id","icu_id", "patient_id"]) # very very large variance and doesn't effect the predection so it will be dropped for easier diagrams
    '''
)
st.code(
    '''
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    '''
)

st.code(
    '''
    df_clean = df[numeric_cols].dropna()
    '''
)

st.code(
    '''
    # spliting data to large and small values so the graphs doesn't feel clunky
    tiny_values = [col for col in df[numeric_cols].columns if df[col].max() <= 50]
    small_values = [col for col in df[numeric_cols].columns if df[col].max() > 50 and df[col].max() < 500 ]
    large_values = [col for col in df[numeric_cols].columns if df[col].max() > 500]
    '''
)

st.code(
    '''
    plt.figure(figsize=(12, 20))
    sns.boxplot(data=df_clean[tiny_values], orient='h')
    plt.title('Boxplots of tiny Numeric Columns')
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 20))
    sns.boxplot(data=df_clean[small_values], orient='h')
    plt.title('Boxplots of small Numeric Columns')
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 20))
    sns.boxplot(data=df_clean[large_values], orient='h')
    plt.title('Boxplots of large Numeric Columns')
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    '''
)

df_clean = df.drop(columns=["hospital_id","icu_id", "patient_id"]) # very very large variance and doesn't effect the predection so it will be dropped for easier diagrams

numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

df_clean.dropna(inplace=True)
df_clean = df_clean[numeric_cols]
# spliting data to large and small values so the graphs doesn't feel clunky
tiny_values = [col for col in df_clean.columns if df_clean[col].max() <= 50]
small_values = [col for col in df_clean.columns if 50 < df_clean[col].max() < 500]
large_values = [col for col in df_clean.columns if df_clean[col].max() >= 500]

column_groups = {
    "Tiny Numeric Columns": tiny_values,
    "Small Numeric Columns": small_values,
    "Large Numeric Columns": large_values
}
# Dropdown in sidebar or main app
selected_group = st.selectbox("Select column group to plot:", list(column_groups.keys()))

# Get selected columns
selected_columns = column_groups[selected_group]

# Plot
if selected_columns:  # Make sure the list isn't empty
    fig = plt.figure(figsize=(12, 20))
    sns.boxplot(data=df_clean[selected_columns], orient='h')
    plt.title(f'Boxplot of {selected_group}')
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("No columns to plot in this group.")


st.code(
    '''
    numeric_features = ['age', 'bmi', 'height']
    categorical_features = [
        'gender', 'ethnicity', 'elective_surgery', 'icu_admit_source',
        'apache_2_bodysystem', 'apache_3j_bodysystem',
        'diabetes_mellitus', 'hepatic_failure', 'immunosuppression',
        'leukemia', 'lymphoma', 'solid_tumor_with_metastasis'
    ]
    important_features = numeric_features + categorical_features
    df = df[important_features + ['hospital_death']] # Filter dataset
    df.dropna(subset=numeric_features, inplace=True) # Drop rows with missing values in key numeric features
    '''
)

numeric_features = ['age', 'bmi', 'height']
categorical_features = [
    'gender', 'ethnicity', 'elective_surgery', 'icu_admit_source',
    'apache_2_bodysystem', 'apache_3j_bodysystem',
    'diabetes_mellitus', 'hepatic_failure', 'immunosuppression',
    'leukemia', 'lymphoma', 'solid_tumor_with_metastasis'
]
important_features = numeric_features + categorical_features
df = df[important_features + ['hospital_death']] # Filter dataset
df.dropna(subset=numeric_features, inplace=True) # Drop rows with missing values in key numeric features

st.subheader('2.6 Numeric Feature Analysis')

st.code(
    '''
    for col in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.violinplot(x='hospital_death', y=col, data=df)  
    plt.title(f'{col} vs Hospital Death (Violin)')
    plt.show()
    '''
)

col1, col2 = st.columns(2)
for col in numeric_features:
    with col1:
        fig1 = plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Count')
        st.pyplot(fig1)

    with col2:
        fig2 = plt.figure(figsize=(8, 4))
        sns.violinplot(x='hospital_death', y=col, data=df)
        plt.title(f'{col} vs Hospital Death (Violin)')
        st.pyplot(fig2)
    

st.code(
    '''
    sns.pairplot(df[numeric_features + ['hospital_death']], hue='hospital_death', diag_kind='kde')
plt.figure(figsize=(8, 4))
plt.suptitle("Pairplot of Numeric Features vs Hospital Death", y=1.02)
plt.show()
'''
)



st.code(
    '''
    plt.figure(figsize=(20, 10))
sns.heatmap(df[numeric_features + ['hospital_death']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
'''
)

plt.figure(figsize=(20, 10))
sns.heatmap(df[numeric_features + ['hospital_death']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
st.pyplot(plt)

st.subheader("2.7 Categorical Feature Analysis")
st.code(
    '''
    for col in categorical_features:
    # 1. Crosstab analysis between categorical feature and hospital_death
    crosstab_result = pd.crosstab(df[col], df['hospital_death'], normalize='index') * 100
    print(f"\nCrosstab analysis for {col} vs Hospital Death:")
    print(crosstab_result)

    # 2. Countplot with hospital_death hue to visualize its relationship with the categorical feature
    plt.figure(figsize=(8, 4))
    x= df["hospital_death"].astype(str)
    sns.countplot(x=col, hue=x, data=df, order=df[col].value_counts().index)
    plt.title(f'{col} vs Hospital Death')
    plt.xticks(rotation=45)
    plt.show()
    '''
)

for col in categorical_features:
    # 1. Crosstab analysis between categorical feature and hospital_death
    crosstab_result = pd.crosstab(df[col], df['hospital_death'], normalize='index') * 100
    st.text(f"Crosstab analysis for {col} vs Hospital Death:")
    st.text(crosstab_result)
    
    # 2. Countplot with hospital_death hue to visualize its relationship with the categorical feature
    plt.figure(figsize=(8, 4))
    x= df["hospital_death"].astype(str)
    sns.countplot(x=col, hue=x, data=df, order=df[col].value_counts().index)
    plt.title(f'{col} vs Hospital Death')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.divider()