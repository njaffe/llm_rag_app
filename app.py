import dotenv
import os
dotenv.load_dotenv()

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage

app = Flask(__name__)

# Load the configuration from environment variables
DATA_FOLDER = os.getenv("DATA_FOLDER", "data/")  # default to "data/" if not specified
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Assuming you have set the OpenAI API key in your .env file

# Initialize the vector store once (for reuse)
def get_vector_store():
    embedding_function = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="resume_collection",
        embedding_function=embedding_function
    )

    return vector_store

# Load documents into the vector store
def load_documents_to_db(document_names):
    vector_store = get_vector_store()

    documents = []
    ids = []

    for idx, doc_name in enumerate(document_names):
        with open(doc_name, "r") as file:
            doc_content = file.read()
        
        document = Document(page_content=doc_content, page_number=idx + 1)
        documents.append(document)
        ids.append(str(idx + 1))

    vector_store.add_documents(documents=documents, ids=ids)

    return vector_store

# Define the RAG query handler
def handle_query(query: str, document_names: list, model="gpt-3.5-turbo"):
    vector_store = load_documents_to_db(document_names)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    docs = retriever.invoke(query)

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. 
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Use the OpenAI API key to instantiate the LLM
    llm = ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY)

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=question_answering_prompt)

    answer = document_chain.invoke(
        {
            "context": docs,
            "messages": [
                HumanMessage(content=query)
            ],
        }
    )

    return answer

# Route for answering questions
@app.route("/ask", methods=["POST"])
def ask_question():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Get all .txt files in the data folder (loaded from the environment)
    document_names = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith(".txt")]
    
    # Get the answer from the RAG model
    answer = handle_query(query=query, document_names=document_names)
    
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=6000)

# curl -X POST http://0.0.0.0:6000/ask -H "Content-Type: application/json" -d '{"query": "What is Noahs work experience with Levis?"}'

# lsof -i :6000
# kill -9 PID
# gunicorn -w 4 -b 0.0.0.0:6000 app:app &
# http://34.71.193.4:6000/app


# gunicorn -w 4 -k gevent --worker-connections 100 -t 600 -b 0.0.0.0:6000 app:app & #dont use this one


# conda activate llm_rag_env