from openai import OpenAI
from tool_use_pattern import ToolUsePattern

client = OpenAI(api_key="sk-...")

def openai_tool_llm(messages, tools):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    ).choices[0].message.to_dict()

# Example tools
def search_web(query: str) -> str:
    """Search the web for a query."""
    return f"Fake search result for '{query}'"

def get_weather(city: str) -> str:
    """Get current weather in a city."""
    return f"{city} is sunny and 28°C."

# Instantiate tool-use agent
tool_agent = ToolUsePattern(
    llm_fn=openai_tool_llm,
    tools={
        "search_web": search_web,
        "get_weather": get_weather
    },
    hooks={
        "after_llm_response": lambda step, resp: print(f"[Step {step}] LLM:", resp.get("function_call") or resp.get("content")),
        "after_tool_use": lambda step, name, args, res: print(f"🛠 {name}({args}) => {res}")
    }
)

res = tool_agent.run("What's the weather in Paris and who is the current president of France?")
print("🧠 Final Output:", res["final_output"])
