# 🐺 Game of Thrones Book QA System (RAG Pipeline)

A robust Retrieval-Augmented Generation (RAG) system using **LangChain (LCEL)**, **OpenAI Embeddings & LLM**, and **FAISS** vector store to answer context-aware questions from the *Game of Thrones* book series.

## 🚀 Key Features

* **Efficient Document Processing**: Merges multiple text corpora seamlessly and chunks text using `RecursiveCharacterTextSplitter`.
* **Fast Local Vector Search**: Leverages `FAISS` with `text-embedding-3-small` embeddings and supports **disk-based local persistence (`save_local`/`load_local`)** to eliminate re-indexing overhead on every run.
* **LangChain Expression Language (LCEL)**: Uses `RunnableParallel`, custom document formatters (`RunnableLambda`), and `StrOutputParser` for clean, modular execution.
* **Context-Aware Responses**: Powered by `gpt-4o-mini` with strict context prompting to prevent hallucinations.

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **Framework**: LangChain (`langchain-core`, `langchain-openai`, `langchain-community`)
* **Vector Store**: FAISS (`faiss-cpu`)
* **LLM & Embeddings**: OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
* **Environment Management**: `python-dotenv`