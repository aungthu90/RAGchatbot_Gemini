import pdfplumber as pdfplumber
import streamlit as st
from pdfplumber import PDF

st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract contents from the PDF and chunk it
if file is not None:
    #extract text from it
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text+= page.extract_text() + "\n"
    st.write(text)
