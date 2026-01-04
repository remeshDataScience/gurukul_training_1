import os
import json
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


print("---completed -----")