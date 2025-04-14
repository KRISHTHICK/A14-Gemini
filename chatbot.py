import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

st.title("🧠 Local PDF Chatbot (No API Key)")

pdf = st.file_uploader("📄 Upload a PDF", type="pdf")

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(splits, embeddings)

    # Load local model
    llm = CTransformers(
        model="models/gemma-2b-it.Q4_K_M.gguf",  # or mistral .gguf path
        model_type="gemma",  # or "mistral"
        config={"max_new_tokens": 256, "temperature": 0.1}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    query = st.text_input("Ask something about the PDF:")
    if query:
        response = qa.run(query)
        st.success(response)
