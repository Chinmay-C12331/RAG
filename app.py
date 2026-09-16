
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG PDF Chatbot")
st.write("Upload a PDF and ask questions about its content.")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to your .env file.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

    st.success(f"PDF processed successfully! {len(chunks)} chunks created.")

    user_query = st.text_input(
        "Ask a question about your PDF:"
    )

    if user_query:

        with st.spinner("Searching and generating answer..."):

            relevant_docs = retriever.invoke(user_query)

            context = "\n\n".join(
                [
                    f"Page: {doc.metadata.get('page', 'Unknown')}\n"
                    f"{doc.page_content}"
                    for doc in relevant_docs
                ]
            )

            prompt = f"""
You are a helpful document-based assistant.

Answer the question using ONLY the provided context.
If the answer is not available in the context,
say that you don't know.

Context:
{context}

Question:
{user_query}
"""

            model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0
            )

            response = model.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        st.subheader("Sources")

        for doc in relevant_docs:
            page_number = doc.metadata.get("page", "Unknown")
            st.write(f"📄 Page {page_number + 1}")
