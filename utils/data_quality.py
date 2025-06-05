# data_quality.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fitz  # pymupdf
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from heapq import nlargest
from navbar import navbar
import io
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load CSS and Navbar
navbar()

# Sidebar for navigation
page = st.sidebar.radio("Go to", ["Home", "Summarise Document", "Data Quality Assessment",
                                  "Exploratory Data Analysis(EDA)", "ML Features", "Results and Recommendations"])

if page == "Home":
    st.title("Welcome to the DQ'Insights tool")
    # Home page content remains unchanged...

elif page == "Data Quality Assessment":
    st.title("Upload CSV Files")

    st.header("This page allows you to upload a CSV file of your choice and conduct data quality checks")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("DataFrame:")
        st.write(data)

        st.header("Data Quality Checks")

        # Function to generate descriptive analysis
        def descriptive_analysis(df):
            st.subheader("Descriptive Analysis")
            st.write(df.describe())

        # Improved Outlier Detection and Capping Function
        def detect_and_cap_outliers(df, method='capping'):
            outlier_details = {}
            for col in df.select_dtypes(include=[np.number]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not outliers.empty:
                    outlier_details[col] = outliers[[col]]
                    if method == 'capping':
                        # Cap outliers
                        mean = df[col].mean()
                        std_dev = df[col].std()
                        cap_lower = mean - 3 * std_dev
                        cap_upper = mean + 3 * std_dev
                        df[col] = np.where(df[col] < cap_lower, cap_lower, df[col])
                        df[col] = np.where(df[col] > cap_upper, cap_upper, df[col])
                    elif method == 'drop':
                        # Drop outliers
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            return df, outlier_details

        # Improved Data Type Checks to handle numerical representations in categorical columns
        def check_data_type_errors(df):
            dtype_errors = {}
            for col in df.columns:
                if col.lower() in ['day', 'month']:  # Recognize day and month as valid categorical data
                    if col.lower() == 'day' and df[col].between(1, 31).all():
                        continue
                    elif col.lower() == 'month' and df[col].between(1, 12).all():
                        continue
                expected_dtype = df[col].dtype
                incorrect_types = df[~df[col].apply(lambda x: isinstance(x, expected_dtype.type))]
                if not incorrect_types.empty:
                    dtype_errors[col] = incorrect_types[[col]]
            return dtype_errors

        # Display Descriptive Analysis
        descriptive_analysis(data)

        # Display Missing Data
        st.subheader("Missing Data")
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            st.write(missing_data)
        else:
            st.write("No missing data detected.")

        # Display Outliers
        st.subheader("Outliers")
        st.write("Select how to handle outliers:")
        outlier_method = st.selectbox("Outlier Handling Method", ["None", "Capping", "Drop"])

        if outlier_method != "None":
            data, outliers = detect_and_cap_outliers(data, method=outlier_method.lower())
            if outliers:
                for col, details in outliers.items():
                    st.write(f"Outliers detected in column: {col}")
                    st.write(details)
                    fig, ax = plt.subplots()
                    ax.boxplot(data[col].dropna())
                    ax.set_title(f"Boxplot for {col} (with potential outliers)")
                    st.pyplot(fig)
            else:
                st.write("No significant outliers detected.")

        # Display Duplications
        st.subheader("Duplications")
        duplicates = data[data.duplicated(keep=False)]
        if not duplicates.empty:
            st.write(f"Found {duplicates.shape[0]} duplicate rows.")
            st.write(duplicates)
        else:
            st.write("No duplicate rows detected.")

        # Display Data Type Errors
        st.subheader("Data Type Errors")
        dtype_errors = check_data_type_errors(data)
        if dtype_errors:
            for col, errors in dtype_errors.items():
                st.write(f"Data type errors in column: {col}")
                st.write(errors)
        else:
            st.write("No data type errors detected.")

        # Download cleaned data if applicable
        if st.button('Download Cleaned Data'):
            cleaned_data = data.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Cleaned CSV", data=cleaned_data, file_name="cleaned_data.csv", mime="text/csv")

elif page == "Exploratory Data Analysis(EDA)":
    st.title("Exploratory Data Analysis (EDA)")

    st.header("Upload CSV Files")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("DataFrame:")
        st.write(data)

        st.header("Data Visualization")

        # Custom Chart Selection
        st.subheader("Choose Chart Type")
        chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Histogram", "Scatter", "Box", "Pie"])

        # Parameter Selection for Custom Chart
        columns = data.columns.tolist()
        x_axis = st.selectbox("Select X-axis column", columns)
        y_axis = st.selectbox("Select Y-axis column", [None] + columns)  # Allow None for single variable charts

        # Display tips on chart types
        st.info("""
        **Chart Type Tips:**
        - **Line**: Best for time series data.
        - **Bar**: Ideal for comparing categories or groups.
        - **Histogram**: Useful for visualizing the distribution of a single variable.
        - **Scatter**: Great for showing the relationship between two continuous variables.
        - **Box**: Good for displaying data distribution and spotting outliers.
        - **Pie**: Works well for displaying proportions of a whole, with categorical data.
        """)

        # Generate the selected chart
        if x_axis:
            st.write(f"{chart_type} chart of {y_axis} vs {x_axis}" if y_axis else f"{chart_type} chart of {x_axis}")

            fig, ax = plt.subplots(figsize=(10, 6))
            try:
                if chart_type == "Line":
                    if y_axis:
                        data.plot(x=x_axis, y=y_axis, ax=ax, kind='line')
                    else:
                        st.error("Line chart requires both X and Y axes.")
                elif chart_type == "Bar":
                    data.plot(x=x_axis, y=y_axis, ax=ax, kind='bar')
                elif chart_type == "Histogram":
                    ax.hist(data[x_axis].dropna(), bins=20)
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel('Frequency')
                elif chart_type == "Scatter":
                    if y_axis:
                        sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
                    else:
                        st.error("Scatter plot requires both X and Y axes.")
                elif chart_type == "Box":
                    sns.boxplot(data=data, x=x_axis, y=y_axis if y_axis else None, ax=ax)
                elif chart_type == "Pie":
                    if y_axis is None:
                        data[x_axis].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                        ax.set_ylabel('')
                    else:
                        st.error("Pie chart requires only one variable.")

                st.pyplot(fig)

                # Button to save the customized figure
                if st.button('Save Chart as PNG'):
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png")
                    buffer.seek(0)
                    st.download_button(label="Download Chart", data=buffer, file_name="chart.png", mime="image/png")
            except Exception as e:
                st.error(f"Error generating chart: {e}")
