from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS to handle cross-origin requests
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize model and embeddings
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
loader = PyPDFDirectoryLoader("./dataset")
docs = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# Create vector database
vectors = FAISS.from_documents(split_docs, embeddings)

@app.route("/proxy/get_answer", methods=["POST", "OPTIONS"])
def get_answer():
    if request.method == "OPTIONS":
        # Handling preflight request
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    try:
        # Get user query from the request
        data = request.json
        user_prompt = data.get("question", "")

        if not user_prompt:
            return jsonify({"error": "Question is required"}), 400

        # Retrieve relevant documents
        retriever = vectors.as_retriever()
        context_docs = retriever.get_relevant_documents(user_prompt)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # Generate response using LLM
        response = llm.invoke(f"As a Financial Advisor, answer in the style of Ben Carlson: {user_prompt} \n\nContext:\n{context_text}")

        return jsonify({"answer": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5020))  # Get the port from the environment variable
    app.run(host="0.0.0.0", port=port)
