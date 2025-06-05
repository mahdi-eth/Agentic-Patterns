from typing import Dict, List
from openai import OpenAI

from reflection_pattern import ReflectionPattern

client = OpenAI(api_key="sk-...")

def openai_chat(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content.strip()

agent = ReflectionPattern(
    generate_fn=openai_chat,
    reflect_fn=openai_chat,
    improve_fn=openai_chat,
    hooks={
        "after_generate": lambda hist, out: print("✅ Initial:", out[:60]),
        "after_reflection": lambda hist, out: print("🔍 Reflection:", out[:60]),
        "after_improve": lambda hist, out: print("🚀 Improved:", out[:60]),
    }
)

result = agent.run("Explain Gödel's incompleteness theorem simply.")
print("💡 Final Answer:\n", result["improved"])
