# Building a raw agent code (using LangChain objects) to get product price after applying
# bronze/silver/gold discount. This code also manually uses langsmith tracing and instrumentation

from dotenv import load_dotenv

load_dotenv()

# To initialise LangChain's init_chat_model function
from langchain.chat_models import init_chat_model

# Transform python functions into custom tools which can be invoked by the agent
from langchain.tools import tool

# Import LangChain Objects: Here HumanMessage - is for user Input.  SystemMessage - To imply our
# message as SystemMessage to LLM, ToolMessage is going to contain the tool result
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# from langchain_deepseek import ChatDeepSeek
# Import langsmit's traceable function, and use it as decorator around functions that we use
from langsmith import traceable
from langchain_openai import ChatOpenAI

# To limit no. of agent execution runs (to 10)
MAX_ITERATIONS = 10
MODEL = "qwen3.5:4b"
# MODEL="qwen2.5:7b"
# --- Tools (LanChain @tool decorator) to convert python function into LangChain tool object,
# automatically generating a schema from function's name, type hints and docString,
# for agent to consume ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"Executing get_product_price(product='{product}')")

    # Define a dictory for product with random values
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 39.65}
    # Below statement return the price of given product, and if the product does not exists
    # in the dictionary, it will return 0 as default
    return prices.get(product, 0)


# Define another tool to get the discounted price, with docString (""" """)
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    # Get the discount % for the given discount_tier, return 0 if it is not available or empty
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Define Agent Loop ---
# run_agent takes question (str) as input parameter
@traceable(name="langchain Agent Loop")
def run_agent(question: str):
    # Get all the tools needed for the agent/llm
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    # For Ollama model using qwen
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    # For DeepSeek, but need credits to call teh api
    # llm = ChatDeepSeek(
    #     model="deepseek-chat",
    #     temperature=0,
    #     max_tokens=200,
    #     timeout=None,
    #     max_retries=2,
    #     # other params...
    # )

    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",   # or gpt-4o, gpt-4.1, gpt-3.5-turbo, etc.
    #     temperature=0
    # )

    # Use langchain's bind_tools method to bind the llm and the tool kit(get_product_price, apply_discount)
    # this bind_tool supports every LLM model that allows tool/function calling
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 70)

    # define the list of messages SystemMessage, with defensive prompts (via STRICT RULES),
    # and user input that will be sent to the LLM
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to product catalog, and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received a price "
                "from get_product_price. Pass the exact price returned by "
                "apply_discount - do NOT pass a made-up number.\n "
                "3. NEVER calculate discount by yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # Iterate LLM
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        # This ai_message will contain either a tool call decision of the LLM
        # or the content if LLM doesn't want to execute tools

        ai_message = llm_with_tools.invoke(messages)
        # Check the tool calls, if we don't receive any calls that means,
        # LLM does not think it requirs to run the tools,
        # if tool calls are empty then print the final answer

        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        # Now process only the FIRST tool call - force once tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")

        print(f"Tool Selected '{tool_name}', id: {tool_id} with args: {tool_args}")
        # Get the tool name using the tool dictionary
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError("Tool '{tool_name}' not found...")

        observation = tool_to_use.invoke(tool_args)
        print(f"[Tool results ] {observation} ")
        # ✅ Append ai message and tool result (observation) back to messages,
        # so that we feedback all the previous steps and observation back to the LLM
        # This creates agentic capability
        messages.append(ai_message)
        messages.append(
            ToolMessage(name=tool_name, content=str(observation), tool_call_id=tool_id)
        )

    # If we exit the loop without final answer:
    print("ERROR: Max Iterations reached without a final answer")
    return None


# Boilerplate codes to run this file
if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    # print a new line
    print()
    # now execute the run_agent fuction with a prompt/question
    result = run_agent("What is the price of a headphones after applying gold discount")
    print(f"Discounted price is '{result}'")
