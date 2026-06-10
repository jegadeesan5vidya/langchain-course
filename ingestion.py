import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    #print(f"PineCone Key {os.environ['PINECONE_API_KEY']}")
    loader = TextLoader("/home/jegan/langchain-course/project/rag-gist/mediumblog1.txt")

    # Now load the file content into LangChain Document using load method of loader
    # Big advantage of using load() abstraction is, it helps to load any source for e.g., 
    # WhatsApp messages, Notion for notion notebooks, google drive.  LangChaing takes care of 
    # loading machanism and keep the interface agnostic
    document = loader.load()
    print(document)

    print("Splitting...")
    # Set Chunk size limit to 1000 characters (based on heuristic, keep it small to fit into the 
    # llm's context window as it may have more than one chunks), chunk_overlap=0 this is to say all 
    # the chunks created, are not going to have overlapping data   
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document) # document contains list (one or more) of langchain document
    print(f"created {len(texts)} chunks")

    #Now load embeddings using openAI
    print("Loading embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    print("Ingesting to PineCone...")
    #What langchain does is, it takes the list of documents/chunks (texts) and creates embeddings for each of the document and 
    # then ingests to pinecone.  It also creates the index if it does not exist.  
    # You can also create the index separately and then ingest the data.  
    # But this is more convenient way to do it in one step.
    # What Langchain offers the flexibility - to use any vector store, you can use 
    # FAISS, Weaviate, Redis etc.  You just need to change the vector store class 
    # and provide the necessary parameters.

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("Ingestion complete!")