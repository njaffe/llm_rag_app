# LLM RAG App

This is a Flask-based web application that leverages a Retrieval-Augmented Generation (RAG) approach to answer questions based on documents stored in a specified directory. The application uses OpenAI’s GPT-3.5 model combined with a Chroma vector store to retrieve relevant information from documents and generate answers.

## Key Features:

- Document Ingestion: Load .txt documents into the Chroma vector store.
- Retrieval: The model retrieves relevant information from documents based on the query provided.
- Generation: The LLM (language model) generates an answer based on the context retrieved.
- Flask API: A simple Flask API endpoint to handle user queries.

## Technologies:

- **Flask**: Lightweight web framework for Python.
- **Gunicorn**: Python WSGI HTTP server for UNIX that serves the Flask app.
- **LangChain**: Framework for building applications using LLMs, vector stores, and document retrieval.
- **Chroma**: Vector database used for storing and retrieving document embeddings.
- **GPT-3.5**: Our LLM used for question answering based on retrieved documents.

## Setup

### Prerequisites:

1.	Python 3.8+ installed.
2.	A .env file containing your OpenAI API key.
3.	A directory (data/) containing .txt documents for the model to process.

### Installation:

1.	Clone this repository:

`git clone https://github.com/your-username/llm_rag_app.git`

`cd llm_rag_app`

2.	Install the required dependencies:

`pip install -r requirements.txt`

3.	Create a .env file in the root of the project and add your OpenAI API key:

`OPENAI_API_KEY=your-openai-api-key`

`DATA_FOLDER=data/`

4.	Ensure you have some .txt files in the data/ directory.

## Usage

### Running the Application:

To start the Flask app, run:

`gunicorn -w 4 -b 0.0.0.0:6000 app:app &`

Then submit a request using:

`curl -X POST http://0.0.0.0:6000/ask -H "Content-Type: application/json" -d '{"query": "<Your question>"}'`

## Notes:

- The data/ directory is where the text documents are stored. The app loads all .txt files in this directory when responding to queries.
- The RAG model first retrieves the most relevant document based on the query, and then the OpenAI model generates an answer based on the context.

## Troubleshooting:

- If you encounter an error such as OPENAI_API_KEY not found, ensure that your .env file is correctly configured and located in the root of the repository.
- If there are no documents in the data/ folder, the app will not be able to generate answers. Make sure your data/ folder contains .txt files.
