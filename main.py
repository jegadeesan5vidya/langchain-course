from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()
import os


def main():
    print("Hello from langchain-course!")
    print("OPEN --> " + os.environ.get("OPENAI_API_KEY"))
    print("GIT --> " + os.environ.get("GIT_KEY"))

    information = """
        Chandrasekaran Joseph Vijay (born 22 June 1974) is an Indian politician and former actor who is currently serving as the ninth chief minister of Tamil Nadu since May 2026. He is the founder and president of the political party Tamilaga Vettri Kazhagam (TVK). Prior to entering politics, he was a leading actor in Tamil cinema and among the highest-paid actors in India, having won numerous accolades including multiple Tamil Nadu State Film Awards and South Indian International Movie Awards.

    Following a stint as a child actor in the 1980s, Vijay made his debut as a leading man in Naalaiya Theerpu (1992), directed by his father S. A. Chandrasekhar. He rose to fame with romance films such as Poove Unakkaga (1996) and Kadhalukku Mariyadhai (1997), Thullatha Manamum Thullum (1999), Kushi (2000) before transitioning into an action star with Thirumalai (2003), Ghilli (2004), Thirupaachi (2005) and Pokkiri (2007). From the 2010s onward, he starred in major commercial successes—including Thuppakki (2012), Kaththi (2014), Theri (2016), Mersal (2017), Master (2021), Leo (2023) and The Greatest of All Time (2024)—which rank among the highest-grossing Tamil films.

    Beyond his film career, Vijay cultivated fan following, which he consolidated into the social welfare organisation Vijay Makkal Iyakkam (VMI) in 2009. Through VMI, he engaged in extensive philanthropic activities across Tamil Nadu. In many of his later films, such as Kaththi (2014) and Sarkar (2018), he cultivated the image of an angry protagonist fighting against corruption, injustice, and social inequality, which frequently brought him into conflict with the state's ruling political establishment.[1]

    In February 2024, Vijay announced his retirement from cinema and his formal entry into politics. He launched the TVK, a secular alternative to the established Dravidian parties. In a watershed moment for Tamil Nadu politics, the TVK secured 108 seats in the 2026 Tamil Nadu Legislative Assembly election, emerging as the single-largest party and breaking the half-century-long electoral duopoly of the DMK and the AIADMK.[2][3] With the support of several smaller allied parties, Vijay was sworn in as chief minister on 10 May 2026.
    """
    # Here value for {information} will be injected during run time 
    summary_template = """
        Given the information {information} about a person, I want you to create:
        1. A short summary
        2. Three interesting facts about that person
    """
    # We can use PromptTemplate
    summary_template = PromptTemplate(
        input_variables = ["information"], template=summary_template 
    )

    #temperature between 0 to 0.3 will make sure the response is deterministic, fatual and probably repeatable
    # Its is good for summarization, for code, for instructions, for test
    # High temperature values i.e., above 0.8, response will be creative, 
    # good for poetry, fiction and out-of-the-box ideas 
    
    #ChatGpt model
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    
    # Use Gemma model
    llm = ChatOllama(temperature=0, model="gemma4:e4b")

    # Define variable chain to get response using summary_template | llm 
    # below command is using Langchain expression language (LCEL) i.e., we create a chain composing 
    # two components ( a prompt template and a large langauge model) 
    # The pipe operator creates a new runnable chain by connecting the output of the left component 
    # to the right component 
    # so here llm (ChatOpenAI) takes the prompt as input and gives the text response
    chain = summary_template | llm
    response = chain.invoke(input={"information":information})

    print(response.content)

if __name__ == "__main__":
    main()
