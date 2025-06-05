# file_handler.py

import pandas as pd
import chardet

def detect_encoding(file):
    """Detect the encoding of a file using chardet."""
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    return result['encoding']

def handle_file_upload(uploaded_file):
    """Handle the uploaded file and load it into a DataFrame."""
    encoding = detect_encoding(uploaded_file)
    uploaded_file.seek(0)  # Reset file pointer to the beginning
    
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e:
            return None
    return df
