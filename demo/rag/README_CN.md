# 使用 seekdb 快速构建 RAG

本教程将引导您把 Markdown 文档导入 seekdb，构建向量检索知识库并通过 Streamlit 启动 RAG 界面。

## 前提条件

- 已安装 Python 3.11 或以上版本
- 已安装 uv
- 已准备好 LLM API Key

## 准备工作

### 1. 设置环境

#### 安装依赖

**基础安装（适用于 `default` 或 `api` embedding 类型）：**

```bash
uv sync
```

**包含本地模型支持（适用于 `local` embedding 类型）：**

```bash
uv sync --extra local
```

> **提示：** 
> - `local` 额外依赖包含 `sentence-transformers` 及相关依赖（约 2-3GB）。
> - 如果您在中国大陆，可以使用国内镜像源加速下载：
>   - 基础安装（清华源）：`uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
>   - 基础安装（阿里源）：`uv sync --index-url https://mirrors.aliyun.com/pypi/simple`
>   - 本地模型（清华源）：`uv sync --extra local --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
>   - 本地模型（阿里源）：`uv sync --extra local --index-url https://mirrors.aliyun.com/pypi/simple`

#### 设置环境变量

**步骤一：复制环境变量模板**

```bash
cp .env.example .env
```

**步骤二：编辑 `.env` 文件，设置环境变量**

本系统支持三种 Embedding 函数类型，您可以根据需求选择：

**1. `default`（默认，推荐新手使用）**
- 使用 pyseekdb 自带的 `DefaultEmbeddingFunction`（基于 ONNX）
- 首次使用会自动下载模型，无需配置 API Key
- 适合本地开发和测试

**2. `local`（本地模型）**
- 使用自定义的 sentence-transformers 模型
- 需要安装 `sentence-transformers` 库
- 可配置模型名称和设备（CPU/GPU）

**3. `api`（API 服务）**
- 使用 OpenAI 兼容的 Embedding API（如 DashScope、OpenAI 等）
- 需要配置 API Key 和模型名称
- 适合生产环境

以下使用通义千问作为示例（使用 `api` 类型）：

```env
# Embedding Function 类型：api, local, default
EMBEDDING_FUNCTION_TYPE=api

# LLM 配置（用于生成答案）
OPENAI_API_KEY=sk-your-dashscope-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL_NAME=qwen-plus

# Embedding API 配置（仅在 EMBEDDING_FUNCTION_TYPE=api 时需要）
EMBEDDING_API_KEY=sk-your-dashscope-key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL_NAME=text-embedding-v4

# 本地模型配置（仅在 EMBEDDING_FUNCTION_TYPE=local 时需要）
SENTENCE_TRANSFORMERS_MODEL_NAME=all-mpnet-base-v2
SENTENCE_TRANSFORMERS_DEVICE=cpu

# seekdb 配置
SEEKDB_DIR=./data/seekdb_rag
SEEKDB_NAME=test
COLLECTION_NAME=embeddings
```

**环境变量说明：**

| 变量名                          | 说明                                           | 默认值/示例值                                    | 必需条件                    |
|---------------------------------|------------------------------------------------|--------------------------------------------------|----------------------------|
| EMBEDDING_FUNCTION_TYPE         | Embedding 函数类型                             | `default`（可选：`api`, `local`, `default`）    | 必须设置                    |
| OPENAI_API_KEY                  | LLM API Key（支持 OpenAI、通义千问等兼容服务） | 必须设置                                         | 必须设置（用于生成答案）    |
| OPENAI_BASE_URL                 | LLM API 基础 URL                               | https://dashscope.aliyuncs.com/compatible-mode/v1 | 可选                        |
| OPENAI_MODEL_NAME               | 语言模型名称                                   | qwen-plus                                        | 可选                        |
| EMBEDDING_API_KEY               | Embedding API Key                              | -                                                | `EMBEDDING_FUNCTION_TYPE=api` 时必需 |
| EMBEDDING_BASE_URL              | Embedding API 基础 URL                         | https://dashscope.aliyuncs.com/compatible-mode/v1 | `EMBEDDING_FUNCTION_TYPE=api` 时可选 |
| EMBEDDING_MODEL_NAME            | Embedding 模型名称                             | text-embedding-v4                                | `EMBEDDING_FUNCTION_TYPE=api` 时必需 |
| SENTENCE_TRANSFORMERS_MODEL_NAME| 本地模型名称                                   | all-mpnet-base-v2                               | `EMBEDDING_FUNCTION_TYPE=local` 时可选 |
| SENTENCE_TRANSFORMERS_DEVICE    | 运行设备                                       | cpu                                              | `EMBEDDING_FUNCTION_TYPE=local` 时可选 |
| SEEKDB_DIR                      | seekdb 数据库目录                              | ./data/seekdb_rag                                | 可选                        |
| SEEKDB_NAME                     | 数据库名称                                     | test                                             | 可选                        |
| COLLECTION_NAME                 | 嵌入表名称                                     | embeddings                                       | 可选                        |

> **提示：** 
> - 如果使用 `default` 类型，只需配置 `EMBEDDING_FUNCTION_TYPE=default` 和 LLM 相关配置即可
> - 如果使用 `api` 类型，需要额外配置 Embedding API 相关变量
> - 如果使用 `local` 类型，需要安装 `sentence-transformers` 库，并可选择配置模型名称

### 2. 准备数据

我们使用 pyseekdb 的 SDK 文档作为示例，您也可以使用自己的 Markdown 文档或者目录

**导入数据：**

运行数据导入脚本：

```bash
# 导入单个文件
uv run python seekdb_insert.py ../../README.md

# 或导入目录下的所有 Markdown 文件
uv run python seekdb_insert.py path/to/your_dir
```

**导入说明：**

在此步骤中，系统会执行如下操作：

- 读取指定的 Markdown 文件或目录下的所有 Markdown 文件
- 将文档按标题分割成文本块（使用 `# ` 分隔符）
- 根据 `.env` 中配置的 `EMBEDDING_FUNCTION_TYPE` 选择相应的 embedding 函数：
  - `default`: 使用 pyseekdb 自带的 `DefaultEmbeddingFunction`（首次使用会自动下载模型）
  - `local`: 使用自定义的 sentence-transformers 模型
  - `api`: 使用配置的 Embedding API 服务
- 自动生成文本嵌入向量
- 将嵌入向量存储到 seekdb 数据库
- 自动跳过失败的文档块，确保批量处理的稳定性

## 构建 RAG

通过 Streamlit 启动应用：

```bash
uv run streamlit run seekdb_app.py
```

启动后，您可以在浏览器中访问 RAG 界面，查询您要检索的数据了。

> **提示：** 使用 `uv` 包管理器，请使用 `uv run` 前缀来运行命令，以确保使用正确的 Python 环境和依赖。
