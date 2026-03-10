
from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Import your LangChain components
from src.helper import download_embadding
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from langchain_classic.chains import  create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI



app = Flask(__name__)


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


## loading embadding from pinecone

embaddings = download_embadding()
index_name = "medicalchatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding = embaddings
)



retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs ={"k":3})


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GEMINI_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{input}"),
        
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)