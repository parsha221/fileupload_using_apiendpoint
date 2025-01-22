import os
import ollama
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
 
import chromadb
import sqlite3
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
class clsManager:
        _instance = None
 
        @staticmethod
        def get_instance():
            if clsManager._instance == None:
                clsManager._instance = clsManager()
            else:
                print("Instance Found")
 
            return clsManager._instance
 
        def __new__(cls,*args, **kwargs):
           if not cls._instance:
              cls._instance = super().__new__(cls)
           return cls._instance
 
        def __init__(self):
            self.query_engine_dic = {};
 
        def run_ingest(self, path, filename):
            try:
                loader = PyPDFLoader(path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                        separators="\n",
                        chunk_size=100000,
                        chunk_overlap=150,
                        length_function=len
                )
 
                all_splits = text_splitter.split_documents(documents)
                vector_store_path = os.path.join('vectorstore', filename)
                vectorstore = Chroma.from_documents(
                    documents = all_splits,
                    collection_name = "rag-chroma",
                    embedding = OllamaEmbeddings(model = 'nomic-embed-text'),
                    persist_directory = vector_store_path,
                )
 
                index = vectorstore.as_retriever()
                self.query_engine_dic[filename] = index
                
        except Exception as e:
                print(e)
 
        def getResponse(self, filename):
            index = self.query_engine_dic[filename]
            return index
