import os
import logging
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# LOGGING CONFIGURATION
# ----------------------------
# Show debug information for LangChain
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# CONFIGURATION
# ----------------------------
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]  # or set via env var

st.header("My First Chatbot (Gemini Embeddings + Logging Test)")

with st.sidebar:
    st.title("Upload Your PDF")
    file = st.file_uploader("Upload a PDF to test embeddings", type="pdf")

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ----------------------------
# MAIN LOGIC
# ----------------------------
if file is not None:
    text = extract_text_from_pdf(file)

    if not text.strip():
        st.error("❌ No extractable text found in this PDF")
    else:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        st.success(f"✂️ Split into {len(chunks)} chunks")
        st.write(chunks[:3])  # preview first 3 chunks

        # Generate embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        # Store embeddings in FAISS
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Setup retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4}
        )

        # ----------------------------
        # LLM (Gemini 1.5 Flash)
        # ----------------------------
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=1000,
            google_api_key=GOOGLE_API_KEY,
            api_version="v1beta"
        )

        # ----------------------------
        # Prompt
        # ----------------------------
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant answering questions about a PDF document.\n\n"
             "Guidelines:\n"
             "1. Provide complete, well-explained answers using the context below.\n"
             "2. Include relevant details, numbers, and explanations.\n"
             "3. Include related info from the context if relevant.\n"
             "4. Only use information from the provided context.\n"
             "5. Summarize long info, use bullets if needed.\n"
             "6. If info is missing, say so politely.\n\n"
             "Context:\n{context}"),
            ("human", "{question}")
        ])

        # ----------------------------
        # RAG CHAIN
        # ----------------------------
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Enable per-chain verbose logging

        # ----------------------------
        # User input
        # ----------------------------
        user_question = st.text_input("Ask a question about your PDF")

        if user_question:
            st.info("🟢 Generating answer...")
            try:
                response = chain.invoke(user_question)
                st.success("✅ Answer generated!")
                st.write(response)
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                logger.exception("Error during chain invocation")
