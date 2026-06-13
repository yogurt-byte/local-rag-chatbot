import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path):
    """load all text files from the docs diretory"""
    print(f"Loading documents from{docs_path}...")

    #check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory{docs_path} does not exist. Please create it and add your company files")

    #load all.txt files from the docs directory
    loader=DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding":"UTF-8"}
    )

    documents=loader.load()

    if len(documents)==0:
        raise FileNotFoundError(f"No .txt file were found in {docs_path}. Pls add your company documents")
    
    for i,doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")
    
    return documents


def split_documents(documents,chunk_size=800,chunk_overlap=0):
    """Splits documents into chunks of lower size"""
    print(f"Splitting documents into chunks...")
    
    text_splitter=CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n---Chunk{i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} charecters")
            print(f"Content: ")
            print(f"{chunk.page_content}")
            print("-"*50)

        if len(chunks)>5:
            print(f"\n...and {len(chunks)-5} more chunks")
    return chunks

def create_vector_store(chunks,persist_directory="db/chroma_db"):
    """Create and persist chromaDB vector store"""
    print("creating embeddings and storing in chromaDB...")

    #embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")  # ← changed

    #Create chromaDB vector storage
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    print("---finished creating vector store---")
    print(f"vector store created and saved to {persist_directory}")
    return vectorstore 


def main():
    print("Main Function")
    #1.loading the files
    documents=load_documents(docs_path="docs")

    #2.Chunking the files
    chunks=split_documents(documents)

    #3.Embedding and storing in vector DB
    vectorstore=create_vector_store(chunks)




if __name__ =="__main__":
    main()