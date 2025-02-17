import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set Hugging Face token and embedding model
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Ensures persistence
vectorstore = Chroma(
    client=chroma_client, 
    collection_name="pdf_embeddings",
    embedding_function=embeddings
)

# Set up Streamlit UI
st.title("📄 Conversational RAG with PDF Uploads & Chat History")
st.write("Upload PDFs and chat with their content in real-time.")

# User inputs Groq API key
api_key = st.text_input("🔑 Enter your Groq API key:", type="password")

# Ensure API key is provided
if not api_key:
    st.warning("Please enter your Groq API key to proceed.")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

# Session ID for managing chat history
session_id = st.text_input("🆔 Session ID", value="default_session")

# Initialize session state storage
if "store" not in st.session_state:
    st.session_state.store = {}

# File uploader for PDFs
uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

# Process uploaded PDFs
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf_path = f"./temp_{uploaded_file.name}"
        
        # Save PDF to temporary file
        with open(temp_pdf_path, "wb") as file:
            file.write(uploaded_file.getvalue())

        # Load and extract text from PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        documents.extend(docs)

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Initialize ChromaDB (Persistent mode for storing vectors)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Create a retriever for fetching relevant documents
    retriever = vectorstore.as_retriever()

    # Contextualizing system prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without previous chat history. Do NOT answer the question, "
        "just reformulate it if needed; otherwise, return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer system prompt
    system_prompt = (
        "You are an AI assistant for question-answering tasks. "
        "Use the provided retrieved context to answer the question. "
        "If you don't know the answer, simply state that you don't know. "
        "Limit your answer to three sentences and keep it concise.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Create Question-Answer Chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Function to manage session history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Create conversational RAG chain with session memory
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User input section
    user_input = st.text_input("📝 Ask a question about the uploaded PDFs:")

    if user_input:
        session_history = get_session_history(session_id)

        # Generate response
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        # Display response and chat history
        st.write("🤖 **Assistant:**", response['answer'])
        st.write("📜 **Chat History:**", session_history.messages)

else:
    st.info("Upload at least one PDF to start chatting.")
