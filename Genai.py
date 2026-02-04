import os
import logging
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# 1. CONFIGURATION & LOGGING
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Access your API Key from Streamlit Cloud Secrets
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
DATA_PATH = "data/"

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.header("🤖 Knowledge Base Chatbot (Geminitest_Aung)")

# ----------------------------
# 2. SIDEBAR - FILE STATUS
# ----------------------------
with st.sidebar:
    st.title("📚 Current Documents")
    if os.path.exists(DATA_PATH):
        files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        if files:
            st.success(f"Found {len(files)} PDF(s) in GitHub folder:")
            for f in files:
                st.markdown(f"- {f}")
        else:
            st.warning("⚠️ Folder 'data/' found but it is empty.")
    else:
        st.error("❌ 'data/' folder not found in repository.")

# ----------------------------
# 3. HELPER FUNCTIONS
# ----------------------------
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ----------------------------
# 4. MAIN RAG LOGIC
# ----------------------------
# Check if we have files to process
if os.path.exists(DATA_PATH) and any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
    
    # Use st.cache_resource so the vector store doesn't rebuild on every click
    @st.cache_resource
    def initialize_vector_store():
        # Load all PDFs from the directory
        loader = PyPDFDirectoryLoader(DATA_PATH)
        raw_documents = loader.load()
        
        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(raw_documents)
        
        # Initialize Google Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Create and return FAISS vector store
        return FAISS.from_documents(chunks, embeddings)

    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    # Initialize LLM (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=1000,
        google_api_key=GOOGLE_API_KEY
    )

    # Define Prompt Template
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

    # Build the RAG Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ----------------------------
    # 5. CHAT INTERFACE
    # ----------------------------
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    response = chain.invoke(user_question)
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception("Chain invocation failed")
else:
    st.info("Please add PDF files to your 'data/' folder on GitHub to start chatting.")


