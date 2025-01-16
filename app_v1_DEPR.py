import os
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
# from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI, OpenAIEmbeddings

import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Initialize Flask app
app = Flask(__name__)

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("resume_collection")

# Load resume into vector database
def load_resume_to_db():
    with open("resume.txt", "r") as file:
        resume_content = file.read()
    
    # Create embeddings and store them in ChromaDB
    embeddings = OpenAIEmbeddings()
    collection.add_documents([resume_content], embeddings)

# Function to handle queries using RAG
def handle_query(query):
    embeddings = OpenAIEmbeddings()
    # Retrieve relevant document from ChromaDB
    db = Chroma(collection_name="resume_collection", embedding_function=embeddings)
        
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            temperature=0.2,
            max_tokens=1000),
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    result = qa_chain.invoke(query)
    return result

# Route for answering questions
@app.route("/ask", methods=["POST"])
def ask_question():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Get answer from RAG model
    answer = handle_query(query)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    load_resume_to_db()  # Load resume data into vector database
    app.run(debug=True, host='0.0.0.0', port=6000)



# curl -X POST http://0.0.0.0:6000/ask -H "Content-Type: application/json" -d '{"query": "What is Noahs work experience with Levis?"}'

# lsof -i :6000
# kill -9 PID
# gunicorn -w 4 -b 0.0.0.0:6000 app:app &
# http://34.71.193.4:6000/app


# gunicorn -w 4 -k gevent --worker-connections 100 -t 600 -b 0.0.0.0:6000 app:app & #dont use this one


# conda activate llm_rag_env