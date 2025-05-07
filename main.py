import streamlit as st

pages = {
    "Data preprocessing": [
        st.Page("cleaning.py", title="Data Cleaining"),
        st.Page("eda.py", title="EDA"),
    ],
    "Model Training": [
        st.Page("lda_reduction.py", title="LDA Reduction"),
        st.Page("svm_grid_training.py", title="SVM"),
    ],
    "Try it yourslef": [
        st.Page("model.py", title="LDA and SVM")
    ]
}

pg = st.navigation(pages)
pg.run()