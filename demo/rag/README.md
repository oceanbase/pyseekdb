# Building RAG with seekdb

This tutorial will guide you through importing Markdown documents into seekdb, building a hybrid search knowledge base, and launching a RAG interface via Streamlit.

## Prerequisites

- Python 3.11 or higher installed
- uv package manager installed
- LLM API Key ready

## Setup

### 1. Environment Setup

#### Install Dependencies

**Basic installation (for `default` or `api` embedding types):**

```bash
uv sync
```

**With local embedding support (for `local` embedding type):**

```bash
uv sync --extra local
```

> **Note:** 
> - The `local` extra includes `sentence-transformers` and related dependencies (~2-3GB).
> - If you experience slow download speeds, you can use mirror sources to accelerate:
>   - Basic installation (Tsinghua mirror): `uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
>   - Basic installation (Aliyun mirror): `uv sync --index-url https://mirrors.aliyun.com/pypi/simple`
>   - Local model (Tsinghua mirror): `uv sync --extra local --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
>   - Local model (Aliyun mirror): `uv sync --extra local --index-url https://mirrors.aliyun.com/pypi/simple`

#### Configure Environment Variables

**Step 1: Copy the environment variable template**

```bash
cp .env.example .env
```

**Step 2: Edit the `.env` file and configure environment variables**

The system supports three types of Embedding functions. You can choose based on your needs:

**1. `default` (Default, recommended for beginners)**
- Uses pyseekdb's built-in `DefaultEmbeddingFunction` (based on ONNX)
- Automatically downloads the model on first use, no API Key configuration required
- Suitable for local development and testing

**2. `local` (Local model)**
- Uses custom sentence-transformers models
- Requires installation of `sentence-transformers` library
- Configurable model name and device (CPU/GPU)

**3. `api` (API service)**
- Uses OpenAI-compatible Embedding API (such as DashScope, OpenAI, etc.)
- Requires API Key and model name configuration
- Suitable for production environments

The following example uses Qwen (with `api` type):

```env
# Embedding Function type: api, local, default
EMBEDDING_FUNCTION_TYPE=api

# LLM configuration (for generating answers)
OPENAI_API_KEY=sk-your-dashscope-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL_NAME=qwen-plus

# Embedding API configuration (required only when EMBEDDING_FUNCTION_TYPE=api)
EMBEDDING_API_KEY=sk-your-dashscope-key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL_NAME=text-embedding-v4

# Local model configuration (required only when EMBEDDING_FUNCTION_TYPE=local)
SENTENCE_TRANSFORMERS_MODEL_NAME=all-mpnet-base-v2
SENTENCE_TRANSFORMERS_DEVICE=cpu

# seekdb configuration
SEEKDB_DIR=./data/seekdb_rag
SEEKDB_NAME=test
COLLECTION_NAME=embeddings
```

**Environment Variable Reference:**

| Variable Name                      | Description                                          | Default/Example Value                            | Required Condition                    |
|------------------------------------|------------------------------------------------------|--------------------------------------------------|---------------------------------------|
| EMBEDDING_FUNCTION_TYPE            | Embedding function type                             | `default` (options: `api`, `local`, `default`)   | Required                               |
| OPENAI_API_KEY                     | LLM API Key (supports OpenAI, Qwen, etc.)           | Must be set                                       | Required (for generating answers)      |
| OPENAI_BASE_URL                    | LLM API base URL                                     | https://dashscope.aliyuncs.com/compatible-mode/v1 | Optional                               |
| OPENAI_MODEL_NAME                  | Language model name                                  | qwen-plus                                        | Optional                               |
| EMBEDDING_API_KEY                  | Embedding API Key                                    | -                                                 | Required when `EMBEDDING_FUNCTION_TYPE=api` |
| EMBEDDING_BASE_URL                 | Embedding API base URL                               | https://dashscope.aliyuncs.com/compatible-mode/v1 | Optional when `EMBEDDING_FUNCTION_TYPE=api` |
| EMBEDDING_MODEL_NAME               | Embedding model name                                | text-embedding-v4                                | Required when `EMBEDDING_FUNCTION_TYPE=api` |
| SENTENCE_TRANSFORMERS_MODEL_NAME   | Local model name                                     | all-mpnet-base-v2                               | Optional when `EMBEDDING_FUNCTION_TYPE=local` |
| SENTENCE_TRANSFORMERS_DEVICE       | Device to run on                                     | cpu                                              | Optional when `EMBEDDING_FUNCTION_TYPE=local` |
| SEEKDB_DIR                         | seekdb database directory                           | ./data/seekdb_rag                                | Optional                               |
| SEEKDB_NAME                        | Database name                                        | test                                             | Optional                               |
| COLLECTION_NAME                    | Collection name                                     | embeddings                                       | Optional                               |

> **Tip:** 
> - If using `default` type, only configure `EMBEDDING_FUNCTION_TYPE=default` and LLM-related settings
> - If using `api` type, additional Embedding API variables need to be configured
> - If using `local` type, install the `sentence-transformers` library and optionally configure the model name

### 2. Prepare Data

We use pyseekdb's SDK documentation as an example. You can also use your own Markdown documents or directory.

**Import Data:**

Run the data import script:

```bash
# Import a single file
uv run python seekdb_insert.py ../../README.md

# Or import all markdown files from a directory
uv run python seekdb_insert.py path/to/your_dir
```

**Import Instructions:**

During this step, the system will perform the following operations:

- Read the specified Markdown file or all Markdown files in the directory
- Split documents into text chunks by headers (using `# ` separator)
- Select the appropriate embedding function based on `EMBEDDING_FUNCTION_TYPE` configured in `.env`:
  - `default`: Uses pyseekdb's built-in `DefaultEmbeddingFunction` (automatically downloads model on first use)
  - `local`: Uses custom sentence-transformers model
  - `api`: Uses configured Embedding API service
- Automatically generate text embedding vectors
- Store embedding vectors in seekdb database
- Automatically skip failed document chunks to ensure batch processing stability

## Build RAG

Launch the application via Streamlit:

```bash
uv run streamlit run seekdb_app.py
```

After launching, you can access the RAG interface in your browser to query your data.

> **Tip:** When using the `uv` package manager, use the `uv run` prefix to run commands to ensure the correct Python environment and dependencies are used.

