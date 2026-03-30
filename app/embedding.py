import os
import pickle
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from app.config import VECTOR_STORE_PATH, METADATA_PATH, AWS_REGION, EMBEDDING_MODEL

def get_embeddings():
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return BedrockEmbeddings(client=bedrock, model_id=EMBEDDING_MODEL)

def create_vector_store(chunks, metadatas):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    return vectorstore

def save_vector_store(vectorstore):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_PATH)

def load_vector_store():
    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        embeddings = get_embeddings()
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def save_file_metadata(metadata):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def get_file_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'rb') as f:
            return pickle.load(f)
    return {}
