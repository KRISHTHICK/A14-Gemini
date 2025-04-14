Awesome! Let’s build a simple chatbot using:

✅ Gemma (local model)
✅ PDF upload with RAG
✅ No API key
✅ Works in VS Code & Streamlit

🛠️ What You'll Get:
A chatbot that:

Lets you upload a PDF

Reads and understands its content

Answers questions based on the PDF

Runs locally using Streamlit

No need for API keys

✅ Step-by-Step Instructions
1. 📦 Install These Packages First
In your terminal (with venv activated), run:

bash
Copy
Edit
pip install streamlit langchain pypdf sentence-transformers chromadb ctransformers
ctransformers is used to run Gemma locally (or any GGUF format LLM)

2. 🧠 Download a Local LLM (like Gemma or Mistral)
Go to: https://huggingface.co/TheBloke

Choose a model like:

gemma-2b-it.Q4_K_M.gguf or

mistral-7b-instruct-v0.1.Q4_K_M.gguf

💾 Place the downloaded .gguf file in your project folder (e.g. models/)

3. 🧠 Code: chatbot.py
Here’s a super simple version of the chatbot:

python
Copy
Edit
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
4. 🚀 Run with Streamlit
In terminal:

bash
Copy
Edit
streamlit run chatbot.py
💡 How It Works (Simple Explanation):
Part	What It Does
PyPDFLoader	Reads uploaded PDF
CharacterTextSplitter	Breaks text into chunks
HuggingFaceEmbeddings	Turns text into vectors
Chroma	Saves vectors for searching
CTransformers	Loads Gemma or Mistral (no API key)
RetrievalQA	Combines search + LLM to answer your question
