import os
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

#rom langchain_community.llms import Ollama  # ✅ NEW: Ollama LLM
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama

# 🔐 Load environment variables
load_dotenv()

# 📄 Load CSV
file_path = f"C:/Users/Dell/OneDrive/Documents/python_code/chat_with_csv/test.csv"
loader = CSVLoader(file_path=file_path, encoding="utf-8")
documents = loader.load()

# 🧠 Create embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model)
vector_store = DocArrayInMemorySearch.from_documents(documents, embedding)

# ✅ Use built-in retriever
retriever = vector_store.as_retriever()

# Use Ollama LLaMA 3.2 model
llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434")

# 🧠 Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Please utilize the data from the CSV file to extract relevant information and insights in response to the user's inquiries. 
        The analysis should include identifying patterns, summarizing key statistics, and generating accurate, coherent, and tailored responses to the user's questions. 
        Ensure that the output maintains precision, contextual awareness, and clarity, incorporating explanations. 
        If a question is not directly related to the provided data, kindly indicate that the inquiry is unrelated.

{context}"""
    ),
    ("human", "{input}")
])

# 🔁 Build the document QA chain
# create_stuff_documents_chain(llm, prompt)
# stiff_chain= create_map_reduce_documents_chain(llm, prompt)t
stuff_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, stuff_chain)

# 💬 Sample query
question = "what is average age of female ??"
response = qa_chain.invoke({"input": question})

# 🖨️ Output the answer
print("----------------------------------------\n")
print("Bot:", response["answer"])