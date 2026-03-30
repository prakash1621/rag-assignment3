"""Embedding utilities for AWS Bedrock"""

import boto3
import json
import numpy as np
from typing import List


class BedrockEmbedder:
    def __init__(self, model_id: str = "amazon.titan-embed-text-v1", region: str = "us-east-1"):
        """
        Initialize Bedrock embedder.
        
        Args:
            model_id: Bedrock embedding model ID
            region: AWS region
        """
        self.model_id = model_id
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector as numpy array
        """
        body = json.dumps({"inputText": text})
        
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')
        
        return np.array(embedding)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of embedding vectors
        """
        return [self.embed_text(text) for text in texts]
    
    def __call__(self, text: str) -> np.ndarray:
        """Allow embedder to be called as a function"""
        return self.embed_text(text)


def get_embedder(config: dict) -> BedrockEmbedder:
    """
    Create embedder from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        BedrockEmbedder instance
    """
    aws_config = config.get('aws', {})
    return BedrockEmbedder(
        model_id=aws_config.get('embedding_model', 'amazon.titan-embed-text-v1'),
        region=aws_config.get('region', 'us-east-1')
    )
