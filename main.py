import os
from dotenv import load_dotenv
#For ReAct prompts form
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings   
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print("Initializing components...")
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
llm = ChatOpenAI()
vector_store = PineconeVectorStore(index_name=os.environ['INDEX_NAME'], embedding=embeddings)
# Use this vector store's search capability to find out similar searches/relevant chunks 
# of information from the vector store based on the query
# Below as_retriever method is going to return us vector store with search capability, 
# we can use this retriever to get the relevant chunks of information from the vector 
# store based on the query. 
# Here k is the top k number of top relevant chunks we want to retrieve from the vector store
# i.e., Top K = how many nearest/relevant neighbors you want to return.
vector_store.as_retriever(search_kwargs={"k": 3})

# Now initialize prompt template for ReAct prompting, we are going to use the same prompt template 
# for both question and answer, context is going to be the relevant chunks (augmented) of information retrieved 
# from the vector store based on the query and question is the user query.
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}

    Provide a detailed answer:"""
)

# Now we need to format the relevant chunks/document nicely that returned from 
# the vector store search. Iterate through the list of chunks/documents and 
# extract the page_content and join them together with new line for readability.
# Note: This formated chunks/documents are going to be used as context in the 
# prompt template for ReAct prompting.
def format_docs(chunks):
    """ Format retrieved chunks/documents into single string."""
    return "\n\n".join([chunk.page_content for chunk in chunks])




if __name__ == "__main__":
    print("Retrieving...")

    # query
    query = "what is pinecone in machine learning?"

    # Option 0: Raw invocation without RAG
    print("\n" + "="*70 + "\nRaw LLM invocation without RAG" + "\n" + "="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(f"Raw LLM Result: {result_raw.content}")

    #Option 1: Implement RAG pipeline with retrieved context from vector store

    # LCEL stands for "Language Chain Execution Logic". It is a framework for 
    # building complex language processing pipelines using a chain of language 
    # models and other components. The idea is to break down a complex task into 
    # smaller, manageable steps that can be executed sequentially or in parallel, 
    # allowing for more efficient and effective processing of natural language data.
    def retrieval_chain_without_lcel(query: str):
