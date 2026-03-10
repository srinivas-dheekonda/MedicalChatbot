from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filter_minimal_docs,text_split,download_embadding
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeSparseVectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_core.language_models import ModelProfile

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


extracted_data = load_pdf_files(data = 'data/')
filter_data = filter_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embadding = download_embadding()


pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key,ssl_verify=False)

index_name = "medicalchatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,
        metric="cosine",
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch  = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embadding,
    index_name=index_name
)
    

