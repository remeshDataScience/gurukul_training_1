import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings   # ✅ Ollama embeddings

# 🔐 Load environment variables
load_dotenv()

# 📄 Load and split PDF
pdf_path = r"D:\gurukul_training\vectordb_101.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 🧾 Split documents into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 🧠 Build FAISS index with Ollama embeddings
embedding = OllamaEmbeddings(model="llama3.2:latest", base_url="http://localhost:11434")
faiss_store = FAISS.from_documents(docs, embedding)

# 🔍 Perform similarity search directly
question = "what is  index in vector db??"
results = faiss_store.similarity_search(question, k=4)

# 🖨️ Print retrieved text chunks
print("----------------------------------------\n")
for i, doc in enumerate(results, 1):
    print(f"Result {i}:\n{doc.page_content}\n")
