# Building a raw agent code (without LangChain objects, and function calling, regular expression - to parse input/output)
# This code uses ReAct prompts (https://smith.langchain.com/hub/hwchase17/react?organizationId=0e14522e-670b-4a99-8b71-e8bf8ce90c9b)
# and Agent Scratchpad
# We receive raw response (text format) from LLM without proper JSON format
# For parsing regular expression (built-in python regEx library)
import re

# We also use inspect - to inpsect live objects' metatadata for the functions to use them as tools
import inspect

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
    # LLM pass string by default, so convert this string price into float for below computation. 
    price=float(price)
    return round(price * (1 - discount / 100), 2)

def get_tool_descriptions(tools):
    descriptions = []
    for tool_name, tool_function in tools.items():
        # Since all the functions are traced using langsmith's traceable, so we need to # use "__wrapped__"
        # attribute for each function to get its original function implementation (without the decorator)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        # Now get the metadata, args, return type and docString of the function
        signature = inspect.signature(original_function)
        # get the docstring of the function, if noting then return ""
        docstring = inspect.getdoc(original_function) or ""
        # Now append the function metadata into description list as per this format
        descriptions.append(f"{tool_name}{signature} - {docstring}")
        # Add new line to each description, so that this can be sent to LLM as part of the prompt
    return "\n".join(descriptions)

# Get all the tools needed for the agent/llm
tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}

tools_description = get_tool_descriptions(tools)
# Get the tool names followed by a comma and space, to pass it to the ReAct prompt
tool_names = ", ".join(tools.keys())

# Change #3: Delete two functions' JSON schemas.  Tools now live inside the prompt as plain text
# We derive descriptions of the functions using Inspect. Create a function to get the description
# that receives tools dictionary.  This function will iterate over the tools and for each tool gets
# its metadata, args, return type and docString, format everything as string and returns it.
# We will inject this String into our prompt and send it to the LLM
#

# ReAct prompt using f-String to consume value of the variables inside the string
# Note: {tools_description} the value will be plugged during the build time (only one {})
# and value for this {{question}} will be plugged dynamically during runtime (using two {{}})
react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You must call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have re
ceived a price from get_product_price. Pass the exact price returned by apply_discount - do NOT pass a made-up number.
3. NEVER calculate discount by yourself using math. Always use the apply_discount tool.
4. If the user does not specify discount tier, ask them which tier to use - do NOT assume one.

Answer the following questions as best you can.  You have acccess to the following tools:

{tools_description}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""

# Change 4: Drop tools from Ollama chat.  The LLM has no idea that it is an Agent -
# All agency comes from the above prompt and our regex parsing below
# options - These contains statement for stop arguments, i.e., instructions to LLM
# to stop based on certain observations
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# --- Define Agent Loop, Implementing manual tool calling (RAW) ---
# run_agent takes question (str) as input parameter
@traceable(name="Ollama Agent raw Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 70)

    # Change 5: One prompt string replaces the system/user message split
    prompt = react_prompt.format(question=question)
    # Scratchpad is to store the history of llm output, observations along with tool choices used
    # during every iteration, append this scratchpad with original prompt (i.e., ReAct prompt )
    scratchpad = ""

    # Iterate LLM
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        # Stop token prevents (using options dict) the LLM from generating its own Observations
        # (i.e., Hallucination), We inject the real tool results instead.
        # The Stop token is for llm to stop the output/content after it encounter \nObservation.

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temparature": 0},
        )
        # Response here is ollama response and not LangChain ai response, get the content,
        # remember llm response is just plain text and not ai_message object
        output = response.message.content
        print(f"LLM Output: \n{output}")
        # Now parse this output manually to extract (using regEx) the answer or errors from
        # llm and present it to the user
        print(f"[Parsing] Looking for final answer from LLM output...\n")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        # If match is found, then take from first group and strip away all the white spaces
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"[Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 70)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # Difference 6: Parse tool calls from raw text with regEx - this code is fragile if LLM doesn't follow
        # the given output format, so parsing may go wrong.  This one of the pain point and not production read
        print(f"[Parsing] Looking for Action and Action Input from LLM output: \n")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "[Parsing] ERROR: Could not parse Action/Action Input from the LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        # The input may contains one or more arguments (with , separated), so need to handle this
        # manually, i.e., split/transform it into comma separate list
        tool_input_raw = action_input_match.group(1).strip()
        print(f"Tool Selected '{tool_name}' with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"[Tool executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            # If the tools available, then execute the tool along with the args, convert the output (int/float/any)
            # as string as LLM handles only text and store it into observation
            observation = str(tools[tool_name](*args))

        print(f"[Tool Result] {observation }")

        # Change 7: Scratchpad history is one growing string re-sent every iteration - this replaces
        # the messages.append which used in earlier version
        # This should follow the same format given in the ReAct prompt
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    # If we exit the loop without final answer:
    print("ERROR: Max Iterations reached without a final answer")
    return None


# Boilerplate codes to run this file
if __name__ == "__main__":
    print("Hello Raw ReAct Prompt Agent!")
    # print a new line
    print()
    # now execute the run_agent fuction with a prompt/question
    result = run_agent("What is the price of a laptop after applying gold discount")
    print(f"Discounted price is '{result}'")
