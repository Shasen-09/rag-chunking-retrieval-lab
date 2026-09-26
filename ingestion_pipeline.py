from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Main function")
    docs = Path("docs")

    documents = load_documents(docs)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    embeddings(vector_store)


def load_documents(docs_path):
    print(f"Laoding documnets from {docs_path}...")

    if not docs_path.exists():
        raise FileNotFoundError("Docs folder does not exits")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}

    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocumnet: {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content Preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Splitting documents into chunks...")

    textsplitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = textsplitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were produced.")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n --- Chunk: {i+1}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Content:")
        print(chunk.page_content)
        print("-"*50)
    if len(chunks) > 5:
        print(f"\n ... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB ")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("---Creating Vector Store---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("---Finish Creating vector store---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def embeddings(vector_store):
    data = vector_store.get(
        include=["documents", "metadatas", "embeddings"]
    )

    print("Number of records:", len(data["ids"]))

    print("First ID:", data["ids"][0])
    print("First document:", data["documents"][0])
    print("First metadata:", data["metadatas"][0])

    print("First embedding:")
    print(data["embeddings"][0])

    print("Embedding dimensions:")
    print(len(data["embeddings"][0]))


if __name__ == "__main__":
    main()
