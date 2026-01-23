"""
Embedding function implementations for pyseekdb.

This module provides various embedding function implementations that can be used
with pyseekdb collections.
"""

from .amazon_bedrock_embedding_function import AmazonBedrockEmbeddingFunction
from .cohere_embedding_function import CohereEmbeddingFunction
from .google_vertex_embedding_function import GoogleVertexEmbeddingFunction
from .jina_embedding_function import JinaEmbeddingFunction
from .litellm_base_embedding_function import LiteLLMBaseEmbeddingFunction
from .ollama_embedding_function import OllamaEmbeddingFunction
from .openai_base_embedding_function import OpenAIBaseEmbeddingFunction
from .openai_embedding_function import OpenAIEmbeddingFunction
from .qwen_embedding_function import QwenEmbeddingFunction
from .sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from .siliconflow_embedding_function import SiliconflowEmbeddingFunction
from .tencent_hunyuan_embedding_function import TencentHunyuanEmbeddingFunction
from .voyageai_embedding_function import VoyageaiEmbeddingFunction

__all__ = [
    "AmazonBedrockEmbeddingFunction",
    "CohereEmbeddingFunction",
    "GoogleVertexEmbeddingFunction",
    "JinaEmbeddingFunction",
    "LiteLLMBaseEmbeddingFunction",
    "OllamaEmbeddingFunction",
    "OpenAIBaseEmbeddingFunction",
    "OpenAIEmbeddingFunction",
    "QwenEmbeddingFunction",
    "SentenceTransformerEmbeddingFunction",
    "SiliconflowEmbeddingFunction",
    "TencentHunyuanEmbeddingFunction",
    "VoyageaiEmbeddingFunction",
]
