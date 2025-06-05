from openai import OpenAI
from planning_pattern import PlanningPattern

client = OpenAI(api_key="sk-...")

def openai_llm_fn(messages, tools):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    ).choices[0].message.to_dict()

# Example Tools
def search_web(query: str) -> str:
    return f"Fake search results for '{query}'"

def calculate(expr: str) -> str:
    try:
        return str(eval(expr))
    except:
        return "Invalid expression"

# Agent Instance
planner = PlanningPattern(
    llm_fn=openai_llm_fn,
    tools={
        "search_web": search_web,
        "calculate": calculate
    },
    hooks={
        "after_llm": lambda step, out: print(f"[Step {step}] Reasoning: {out.get('content', '') or out['function_call']}"),
        "after_action": lambda step, tool, args, result: print(f"🛠 {tool}({args}) -> {result}")
    }
)

res = planner.run("What is the square root of the sum of 25 and 144?")
print("🧠 Final Output:", res["final_output"])
