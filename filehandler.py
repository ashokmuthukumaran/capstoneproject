import PyPDF2
import pandas as pd
# -------------------------
# Agent 1: CSV Reader Agent
# -------------------------
class CSVReaderAgent:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def read_data(self):
        return pd.read_csv(self.filepath)
    
# -----------------------
# Helper Functions
# -----------------------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""
    