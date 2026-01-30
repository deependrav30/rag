import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path = 'docs'):
    """Load documents from the specified directory."""
    print(f"Loading documents from {docs_path}")

    #Check if directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    # Load documents using DirectoryLoader
    loader = DirectoryLoader(docs_path, glob='*.txt', loader_cls=TextLoader)
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No .txt file found in {docs_path}. Please add your company documents.")

    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content Preview: {doc.page_content[:100]}...")  # Print first 200 characters 
        print(f" metadata: {doc.metadata}")

    return documents    

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks."""
    print("Splitting documents into chunks")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Content Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)

        if(len(chunks) > 5):
            print(f"\nTotal chunks created: {len(chunks)}. Displaying first 5 chunks only.")

    return chunks

def create_vector_store(chunks, persist_directory='db/chroma_db'):
    """Create a Chroma vector store from document chunks."""
    print("Creating vector store and storing it in chroma db")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Creating Chroma Vector Store ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"} 
    )
    print("--- Finished Creating Chroma Vector Store ---")
    print(f"Vector store created and persisted at {persist_directory}")

    return vector_store    

def main():
    print("Main Function Started")
    # Load environment variables
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # if not openai_api_key:
    #     raise ValueError("OPENAI_API_KEY not found in environment variables")   
    # print("Environment variables loaded successfully")

    # Load documents from the 'docs' directory
    documents = load_documents(docs_path='docs')

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()    