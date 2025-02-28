from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Load and process PDF data
extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()  # Ensure this returns a callable function if needed

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medbot"

# Check if the index exists before creating it
existing_indexes = [index["name"] for index in pc.list_indexes()["indexes"]]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to the existing index
index = pc.Index(name=index_name)

# Embed each chunk and upsert the embeddings into Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,  # Ensure this is a callable function if needed
)
