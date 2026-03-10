from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import  create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import ModelProfile

from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
import os



## Extract Text From PDF Files 

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents



def filter_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document]=[]
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source":src}
            )
        )
    return minimal_docs



def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks



def download_embadding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    embaddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embaddings