"""
Vector store indexer — creates, saves, and loads FAISS vector stores.
"""

import os
import pickle
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
EMBEDDING_MODEL = os.environ.get("AWS_EMBEDDING_MODEL", "amazon.titan-embed-text-v1")


def get_embeddings():
    """Get Bedrock embeddings client."""
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return BedrockEmbeddings(client=bedrock, model_id=EMBEDDING_MODEL)


def create_vector_store(chunks, metadatas):
    """Create FAISS vector store from text chunks."""
    embeddings = get_embeddings()
    return FAISS.from_texts(chunks, embeddings, metadatas=metadatas)


def save_vector_store(vectorstore):
    """Save FAISS vector store to disk."""
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_PATH)


def load_vector_store():
    """Load FAISS vector store from disk."""
    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        embeddings = get_embeddings()
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None


def save_file_metadata(metadata):
    """Save file metadata for change detection."""
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)


def get_file_metadata():
    """Load saved file metadata."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'rb') as f:
            return pickle.load(f)
    return {}
