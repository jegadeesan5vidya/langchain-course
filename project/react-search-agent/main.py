from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field

# Use Pydantic object - to define a structured data scheme. We use two BaseModel which gives functionality like 
# data parsing, serialization, and automatic type validations.  And Field class which help us to add
# Metadata like description to our models as attribute
# 
 

# To create an Agent
from langchain.agents import create_agent
# To give tool to agent
from langchain.tools import tool
# We are using HumanMessage to invoke the Agent
from langchain_core.messages import HumanMessage
# ChatOpenAI
from langchain_openai import ChatOpenAI
#import Tavily
#from tavily import TavilyClient
# Import langchain-tavily
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama

#Initialize tavily
#tavily=TavilyClient()


# Note: Langchain Toolkit provides tools for agent, tools for LLM.  A tool is a function that agent 
# can execute, we can write internal implementation of that function.  for e.g., we can make it to call
# an API to search in a database and execute a code - possibilities are endless
# Once we create a tool, we can plug it into an agent to use that capabilities    
# Below code is defining a custom tool/function, which takes a search string and return results 
# The function should have docString description within """<content>""" which will be used by the agent/LLM to choose which 
# tool/function to invoke based on the docString description.  It should be explicit and unambiguous so that 
# agent can pick when necessary
# The @tool decorator converts this regular python into a linkchain tool 
@tool
def search(query: str) -> str:
    """ 
    Tool that search over internet
    Args: 
        query: The query string to serach for
    Returns: 
        The search result  
    """
    print(f"Searching for {query}")
    # Return default string - dummy search 
    #return "Today's weather is sunny"
    #Search using Tavily and return the result
    return tavily.search(query=query)

# Create the pydantic object, define a new class called Source which inherits from BaseModel
# Give docString description using """ """
# Define class's fields using Field class with description, and this field will be used by LLM
class Source(BaseModel) :
    """ Schema for a source used by an Agent """
    url:str = Field(description="The URL of the source")

# We will nest about pydantic source class/object into another pydantic object called Agentic Response
# This response is the answer from agent which is going to have list of sources (with the source url)
# With answer attribute/field of string type (str)
# sources attribute/field for list of pydantic Source objects
#  
class AgentResponse(BaseModel) :
    """Schema for the Agent response with answers and its sources"""
    answer:str = Field(description = "This is the agent's response to the query")
    # default_factory is list and agent did not provide any response, this would be an empty list
    sources: List[Source] = Field(default_factory=list, description = "List of sources used to generate the Agent's response")

# Now define llm and tools
llm = ChatOpenAI(model="gpt-5")

#llm = ChatOllama(temperature=0.1, model="gemma4:e4b") 

#tools = [search] # Can define n no. of tools similar to search
# Create an agent with llm model and necessary tools to execute/run 

# Using LangChain's Tavily Search
# Initialize Tavily Search, this TavilySearch is already defined as tool using @tool decorator
tools = [TavilySearch()]

# We want agent to return an object instead of string, so that it can be used by downstream system/function, 
# or we can render this response in a front end application etc.
# For this we can add response_format when we create the model
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from react-search-agent!")
    # Now invoke the agent with Human Message to search for weather
    #result = agent.invoke({"messages" : HumanMessage(content = "What is today's weather for Singapore")})
    # With job search query
    result = agent.invoke({"messages" : HumanMessage(content = "Search for latest 2 job postings (application status should not be closed) for an AI engineer using langchain in Singapore job market and list their details. Restrict the search to two or three sources and limit output to 100 tokens.")})
    print(result)

if __name__ == "__main__":
    main()
