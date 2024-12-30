from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
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

@app.route("/get_answer", methods=["POST", "OPTIONS"])
def get_answer():
    if request.method == "OPTIONS":
        # Handling preflight request
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
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

        response = llm.invoke(f"Only answer the questions that is asked, please dont mention the context. Answer the following question using the context below if it is out of context, just provide related answer from the context don't mention that the question is out of context, Start in this way 'As a Financial Advisor'. Answer in the style of Ben Carlson, a financial advisor, and podcaster:\n\n{context_text}\n\nQuery: {user_prompt}")

        return jsonify({"answer": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5020)