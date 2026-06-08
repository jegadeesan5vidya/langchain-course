# Building a raw agent code (without LangChain objects) to get product price after applying
# bronze/silver/gold discount - Raw function calling. This code also manually uses langsmith
# tracing and instrumentation
# ***** Main Differences *****
# 1. LangChain and Ollama use completely different tool‑call formats
# 2. LangChain automatically injects a hidden system prompt
# 3. LangChain automatically formats tool messages
# 4. LangChain automatically handles function invocation
#


from dotenv import load_dotenv

load_dotenv()
# Import ollama python sdk
import ollama
from langsmith import traceable

# To limit no. of agent execution runs (to 10)
MAX_ITERATIONS = 10
MODEL = "qwen3.5:4b"
# MODEL="qwen2.5:7b"


# Trace below function using langsmith as run_type="tool"
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"Executing get_product_price(product='{product}')")

    # Define a dictory for product with random values
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 39.65}
    # Below statement return the price of given product, and if the product does not exists
    # in the dictionary, it will return 0 as default
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    # Get the discount % for the given discount_tier, return 0 if it is not available or empty
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Above two python functions needs to be digest into llm model to use them in function calling as tools
# Refer to ollama tool calling docs @ https://docs.ollama.com/capabilities/tool-calling
# Anthropic tool calling docs @ https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview

# Now define tools. Difference 2: Without @tool, we must MANUALLY define the JSON schema for each
# function. This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and Google docstring format

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The Product name, e.g., 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.


# Now create auxiliary function to add langsmith tracebility manually for ollama calls
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --- Define Agent Loop ---
# run_agent takes question (str) as input parameter
@traceable(name="Ollama Agent Loop (My Ver)")
def run_agent(question: str):
    # Get all the tools needed for the agent/llm
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 70)

    # Ollama Specific messages: Define the list of messages with role system, user with
    # defensive prompts (via STRICT RULES), and user input that will be sent to the LLM.
    # Difference 4: Other models like anthropic or chatgpt, below format will change
    # (LangChain manages this automatically)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant. "
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
            "5. ALWAYS respond using a tool call when tools are available.\n",
        },
        {"role": "user", "content": question},
    ]

    # Iterate LLM
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: Here our function ollama_chat_traced calls ollama.chat() directly
        # and pass the tools, instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        # Response here is ollama response and not LangChain ai response
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content

        # Now process only the FIRST tool call - force once tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"Tool Selected '{tool_name}' with args: {tool_args}")
        # Get the tool name using the tool dictionary
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError("Tool '{tool_name}' not found...")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)
        print(f"[Tool results ] {observation} ")

        # ✅ Append ai message and tool result (observation) back to messages,
        # so that we feedback all the previous steps and observation back to the LLM
        # This creates agentic capability
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    # If we exit the loop without final answer:
    print("ERROR: Max Iterations reached without a final answer")
    return None


# Boilerplate codes to run this file
if __name__ == "__main__":
    print("Hello LangChain Agent (without .bind_tools)!")
    # print a new line
    print()
    # now execute the run_agent fuction with a prompt/question
    result = run_agent("What is the price of a laptop after applying gold discount")
    print(f"Discounted price is '{result}'")
