import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
                                  "Exploratory Data Analysis(EDA)", "ML Features", "Results and Reccommendations"  ])

if page == "Home":
    st.title("Welcome to the DQ'Insights tool")

    st.write("""
    This app allows you to:
    - **Upload and Summarise PDF Files**:
      - Extract and view text from PDF files.
      - Generate a summary of the text.
      - As an example PDF file, we have the Making-Laws.pdf file to experiment with.
    
    - **Perform Data Quality(DQ) Checks of CSV Files**: 
        - **Missing Data**: Identifies null or missing values.
        - **Outliers**: Detects data points that deviate significantly.
        - **Inconsistencies**: Flags mismatched data patterns.
        - **Duplications**: Identifies duplicate records.
        - **Invalid Values**: Detects out-of-range values.
        - **Data Type Errors**: Identifies type mismatches.

    - **Perform Exploratory Data Analysis(EDA) of CSV Files**: 
      - Upload a CSV file and view its contents(200MB limit)
      - See a default line chart of all columns.
      - Customise your chart by selecting columns for X and Y axes.
      - Save your customised charts.
      - As an example CSV file, we have the daily-bike-share.csv file to experiment with.
      
    - **Machine Learning Functionality**: 
      - Apply machine learning with guidance on typical steps and deployment options.

    - **Results and Reccomendations**: 
      - After running the analysis and ML models, you will see results here, along with recommendations 
    for next steps and deployment options.    

    Feel free to use the sidebar to navigate through the different features.
    """)

elif page == "Exploratory Data Analysis(EDA)":
    st.title("Upload CSV Files")

    st.header("This page allows you to upload a CSV file of your choice")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("DataFrame:")
        st.write(data)

        st.header("Data Visualisation")
        
        # Display default chart
        st.write("Here's a simple line chart of the data:")
        fig_default, ax_default = plt.subplots()
        data.plot(ax=ax_default)
        st.pyplot(fig_default)

        # Parameter Selection for custom chart
        st.subheader("Customise your chart")
        columns = data.columns.tolist()
        x_axis = st.selectbox("Select X-axis column", columns)
        y_axis = st.selectbox("Select Y-axis column", columns)

        if x_axis and y_axis:
            st.write(f"Line chart of {y_axis} vs {x_axis}")

            fig_custom, ax_custom = plt.subplots()
            data.plot(x=x_axis, y=y_axis, ax=ax_custom)
            st.pyplot(fig_custom)

            # Button to save the customised figure
            if st.button('Save Chart as PNG'):
                buffer = io.BytesIO()
                fig_custom.savefig(buffer, format="png")  # Save the customised chart
                buffer.seek(0)
                st.download_button(label="Download Chart", data=buffer, file_name="chart.png", mime="image/png")

elif page == "Summarise Document":
    st.title("Upload PDF Files")

    st.header("This page allows you to upload a PDF file of your choice")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from PDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        with st.expander("View Extracted Text"):
            st.write(text)

        # Summarise the text
        def summarise_text(text, num_sentences=5):
            sentences = sent_tokenize(text)
            if len(sentences) < num_sentences:
                return text
            
            stop_words = set(stopwords.words('english'))
            word_frequencies = {}
            for word in word_tokenize(text):
                if word.lower() not in stop_words:
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

            max_frequency = max(word_frequencies.values())
            for word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word] / max_frequency

            sentence_scores = {}
            for sent in sentences:
                for word in word_tokenize(sent.lower()):
                    if word in word_frequencies:
                        if len(sent.split(' ')) < 30:
                            if sent not in sentence_scores:
                                sentence_scores[sent] = word_frequencies[word]
                            else:
                                sentence_scores[sent] += word_frequencies[word]

            summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            summary = ' '.join(summary_sentences)
            return summary

        summary = summarise_text(text, num_sentences=5)

        with st.expander("View Summary"):
            st.write(summary)

        # Download options
        if st.button('Download Extracted Text'):
            st.download_button("Download Text", text, file_name="extracted_text.txt")

        if st.button('Download Summary'):
            st.download_button("Download Summary", summary, file_name="summary.txt")

elif page == "Data Quality Assessment":
    st.title("Upload CSV Files")

    st.header("This page allows you to upload a CSV file of your choice and conduct data quality checks")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("DataFrame:")
        st.write(data)

        st.header("Data Quality Checks")

        # Function to identify missing data
        def check_missing_data(df):
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            return missing_data

        # Function to detect outliers using IQR method and display results
        def detect_outliers(df):
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
            return outlier_details

        # Function to detect duplicates, with an option to check specific columns
        def detect_duplicates(df):
            # Optionally specify subset of columns for duplicate detection
            duplicate_rows = df[df.duplicated(subset=None, keep=False)]
            return duplicate_rows

        # Function to check for data type errors and display problematic rows
        def check_data_type_errors(df):
            dtype_errors = {}
            for col in df.columns:
                expected_dtype = df[col].dtype
                incorrect_types = df[~df[col].apply(lambda x: isinstance(x, expected_dtype.type))]
                if not incorrect_types.empty:
                    dtype_errors[col] = incorrect_types[[col]]
            return dtype_errors

        # Function to detect invalid values
        def detect_invalid_values(df):
            invalid_values = {}
            for col in df.select_dtypes(include=[np.number]):
                invalid_count = df[(df[col] < 0) | (df[col] > 1000000)].shape[0]  # Example rule
                if invalid_count > 0:
                    invalid_values[col] = invalid_count
            return invalid_values

        # Display Missing Data
        st.subheader("Missing Data")
        missing_data = check_missing_data(data)
        if not missing_data.empty:
            st.write(missing_data)
        else:
            st.write("No missing data detected.")

        # Display Outliers
        st.subheader("Outliers")
        outliers = detect_outliers(data)
        if outliers:
            for col, details in outliers.items():
                st.write(f"Outliers detected in column: {col}")
                st.write(details)
                # Optional: Display a box plot for outlier visualization
                fig, ax = plt.subplots()
                ax.boxplot(data[col].dropna())
                ax.set_title(f"Boxplot for {col} (with potential outliers)")
                st.pyplot(fig)
        else:
            st.write("No significant outliers detected.")

        # Display Duplications
        st.subheader("Duplications")
        duplicates = detect_duplicates(data)
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

        # Display Invalid Values
        st.subheader("Invalid Values")
        invalid_values = detect_invalid_values(data)
        if invalid_values:
            st.write(invalid_values)
        else:
            st.write("No invalid values detected.")

elif page == "ML Features":
    st.title("Machine Learning functionalities")

    st.header("This page allows you to upload explore ML features such as data split, data train, data test against ML classifiers and evaluation")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

elif page == "Results and Reccommendations":
    st.title("Results and Reccommendations")

    st.header("After running the analysis and ML models, you will see results here, along with recommendations for next steps and deployment options.")