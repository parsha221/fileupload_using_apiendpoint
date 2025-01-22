import os
import ollama
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
 
import json
import chromadb
import sqlite3
import Manager
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, Blueprint, request, jsonify
 
app = Flask(__name__)
 
queryEngine = ""
path = ""
filename = ""
instance = Manager.clsManager.get_instance()
 
@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Check if the 'file' part is present in the request
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request!'}), 400
 
        file = request.files['file']
        # Check if a file was selected for uploading
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading!'}), 400
 
    except Exception as e:
        print(e)
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500
 
 
    filename = ""
    # Check if the selected file has an allowed file extension
    if file and allowed_file(file.filename):
        filename = file.filename
        path = os.path.join('data', filename)
 
    # Save the file to the specified upload folder
    file.save(path)
    # instance = Manager.clsManager.get_instance()
    instance.run_ingest(path, filename)
 
    return jsonify({'message': 'File uploaded and stored successfully!'}), 200
 
@app.route('/api/ask', methods=['POST'])
def query():
    llm = ChatOllama(model="llama3.2")
    question = request.json['question']
    after_rag_template = """Answer the question based only on the following context:
                            {context}
                            Question: {question}
                         """
 
 
    queryengine = instance.getQueryEngine("abc.pdf")
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": queryengine, "question": RunnablePassthrough()}
        | after_rag_prompt
        | llm
        | StrOutputParser()
    )
    response = after_rag_chain.invoke(question)
 
    return jsonify({'response': response}), 200
 
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in "pdf docx txt xlsx"
 
# Run the Flask application
if __name__ == '__main__':
    # logger.info("Starting the Genesis applciation")
    app.run(debug = True, host = '0.0.0.0', port = '5000')
