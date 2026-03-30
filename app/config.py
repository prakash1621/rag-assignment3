import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge-base")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")

# AWS Bedrock
AWS_REGION = "us-east-1"
EMBEDDING_MODEL = "amazon.titan-embed-text-v1"
LLM_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

# RAG Settings
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
RETRIEVAL_K = 10
RERANK_TOP_K = 3

# Category Keywords
CATEGORY_KEYWORDS = {
    "Teradata": ["teradata", "data discrepancy", "discrepancies", "td", "sql"],
    "pusa-sell-kb": ["pusa", "sell", "seat", "passenger", "res_spcl", "res_dcs", "redshift", "uat"],
    "dot": ["dot", "fare", "currency", "calculation"],
    "avaya": ["avaya", "qa", "validation", "regression", "testing"],
    "swav": ["swav", "vacation", "qmo", "automation"],
    "galaxy": ["galaxy", "currency conversion"],
    "bppsl": ["bppsl", "reference"],
}
