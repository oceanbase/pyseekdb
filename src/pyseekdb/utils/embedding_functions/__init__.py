"""
Embedding function implementations for pyseekdb.

This module provides various embedding function implementations that can be used
with pyseekdb collections.
"""

from .sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from .litellm_base_embedding_function import LiteLLMBaseEmbeddingFunction
from .openai_embedding_function import OpenAIEmbeddingFunction
from .qwen_embedding_function import QwenEmbeddingFunction
from .siliconflow_embedding_function import SiliconflowEmbeddingFunction
from .tengxun_hunyuan_embedding_function import TengxunHunyuanEmbeddingFunction
from .ollama_embedding_function import OllamaEmbeddingFunction
from .voyageai_embedding_function import VoyageaiEmbeddingFunction
from .google_vertex_embedding_function import GoogleVertexEmbeddingFunction
from .cohere_embedding_function import CohereEmbeddingFunction
from .jina_embedding_function import JinaEmbeddingFunction
from .amazon_bedrock_embedding_function import AmazonBedrockEmbeddingFunction

__all__ = [
    "SentenceTransformerEmbeddingFunction",
    "LiteLLMBaseEmbeddingFunction",
    "OpenAIEmbeddingFunction",
    "QwenEmbeddingFunction",
    "SiliconflowEmbeddingFunction",
    "TengxunHunyuanEmbeddingFunction",
    "OllamaEmbeddingFunction",
    "VoyageaiEmbeddingFunction",
    "GoogleVertexEmbeddingFunction",
    "CohereEmbeddingFunction",
    "JinaEmbeddingFunction",
    "AmazonBedrockEmbeddingFunction",
]
