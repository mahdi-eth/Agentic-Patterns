# Define fake tools
def search_web(query: str) -> str:
    return f"Search result for '{query}'"

def calculate(expr: str) -> str:
    return str(eval(expr))

# OpenAI example
from multiagent_pattern.cognitive_agent import CognitiveAgent
from multiagent_pattern.crew import AgentCrew
from openai import OpenAI
client = OpenAI(api_key="sk-...")

def openai_fn(messages, tools):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    ).choices[0].message.to_dict()

# Agents
agent_researcher = CognitiveAgent("researcher", "tool", openai_fn, {"search_web": search_web})
agent_calculator = CognitiveAgent("calculator", "tool", openai_fn, {"calculate": calculate})
agent_reflector = CognitiveAgent("reviewer", "reflection", openai_fn)

# Crew
crew = AgentCrew([agent_researcher, agent_calculator, agent_reflector])

# Run
result = crew.run("What's the square root of (25 + 144) and who is the president of France?")
for name, output in result.items():
    print(f"\n🧠 {name.upper()}:\n", output["output"] if "output" in output else output["improved"])
