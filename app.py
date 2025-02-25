import os
import getpass
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCSU9sqE3nSAzrPpomPjVU2xm9MsWMdu0Y"

# Flask App
app = Flask(__name__)
CORS(app)

# Storage for ChromaDB
DB_PATH = "embeddings"
os.makedirs(DB_PATH, exist_ok=True)

PDF_PATH = "./docs.pdf"  

# Function to process the given PDF
def ingest_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    return text_splitter.split_documents(documents)

# Function to embed documents
def embed_documents(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=DB_PATH)

# Load existing or create new vector store
def load_vectorstore():
    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
    return None

# Ingest and embed the provided PDF
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"Missing PDF file: {PDF_PATH}")

documents = ingest_documents(PDF_PATH)
vectorstore = embed_documents(documents)

# Create retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the retrieved context to answer the question. "
    "If unsure, say 'I don't know'. Keep answers concise."
    "\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("input", "")
    if not query:
        return jsonify({"error": "No input provided"}), 400
    
    response = rag_chain.invoke({"input": query})
    return jsonify({"answer": response["answer"]})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
