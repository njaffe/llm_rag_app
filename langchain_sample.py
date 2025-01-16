import dotenv
dotenv.load_dotenv()

# from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb 

# ---------------------------------------------------------------------------------------------------- 

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

message = "What is Noahs experience with Levis?"


vector_store = Chroma(
    collection_name="resume_collection",
    embedding_function=OpenAIEmbeddings())

from langchain_core.documents import Document

with open("resume.txt", "r") as file:
    resume_content = file.read()

document_1 = Document(page_content=resume_content, page_number=1)

documents = [document_1]
ids = ["1"]
vector_store.add_documents(documents=documents, ids=ids)

# ----------------------------------------------------------------------------------------------------

# Set up the retriever

retriever = vector_store.as_retriever(
    search_kwargs={"k": 1},
)
# docs = retriever.invoke("Describe Noah's resume")
docs = retriever.invoke(message)

# ----------------------------------------------------------------------------------------------------

# Chains

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

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

llm = ChatOpenAI(model="gpt-3.5-turbo")

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt = question_answering_prompt)

from langchain_core.messages import HumanMessage

answer = document_chain.invoke(
    {
        "context": docs,
        "messages": [
            HumanMessage(content=message)
        ],
    }
)

print(answer)