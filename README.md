# My RAG Learning Lab

This is a small project I made to learn the basics of **RAG (Retrieval-Augmented Generation)**.

I am using **Python, LangChain, and ChromaDB** to understand how RAG works step by step.

The project is mainly for learning and experimenting with different ways of splitting text, storing information, searching for useful information, and getting answers from documents.

## What I am learning

### 1. Splitting text

In RAG, large documents are usually divided into smaller pieces called **chunks**.

I have tried different ways of doing this:

* `semantic_chunking.py` - tries to split text based on its meaning.
* `chunking_strategies.py` - tries different methods of splitting text.
* `agentic_chunking.py` - uses an LLM to help decide where the text should be split.

### 2. Storing documents

`ingestion_pipeline.py` takes the files from the `docs/` folder and stores their information in **ChromaDB**.

This helps us search the documents later.

### 3. Searching for information

`retireval_pipeline.py` takes a question and searches the stored documents to find information related to that question.

The retrieved information is then used to generate an answer.

### 4. Chat history

`history_aware_generation.py` is an experiment with conversation history.

The idea is to let the system remember previous questions so it can understand follow-up questions better.

## Project Structure

```text
RAG/
 db/chroma_db/               # Where the database is stored
 docs/                       # My PDF and TXT files
 .env                        # API keys
 agentic_chunking.py         # Experiment with AI-based chunking
 chunking_strategies.py      # Different chunking methods
 history_aware_generation.py # Chat history experiment
 ingestion_pipeline.py      # Adds documents to the database
 retireval_pipeline.py      # Searches for information
 semantic_chunking.py        # Splits text based on meaning
```

## How to run it

First, download the project:

```bash
git clone https://github.com/your-username/rag-chunking-retrieval-lab.git
cd rag-chunking-retrieval-lab
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install langchain langchain-experimental langchain-huggingface chromadb python-dotenv
```

Create a `.env` file and add your API keys:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here

```

Then you can run the Python files:

```bash
python semantic_chunking.py
python ingestion_pipeline.py
python retireval_pipeline.py
```

## Why I made this project

I made this project to learn RAG by actually building small examples instead of only reading about it.

I am still learning, so this repository is mainly a **learning project** where I can try different ideas and understand how each part of a RAG system works.
