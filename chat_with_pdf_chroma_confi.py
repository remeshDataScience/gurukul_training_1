import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
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

# 🧠 Build Chroma index with Ollama embeddings
embedding = OllamaEmbeddings(model="llama3.2:latest", base_url="http://localhost:11434")

# Persist directory for Chroma DB (optional, so you can reload later)
persist_directory = "chroma_db"

chroma_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=persist_directory
)

# 🔍 Perform similarity search with confidence scores
question = "what is Memory Management??"
results = chroma_store.similarity_search_with_score(question, k=4)

# 🖨️ Print retrieved text chunks with scores
print("----------------------------------------\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"Result {i} (score={score/100:.2f}:\n{doc.page_content}\n")
