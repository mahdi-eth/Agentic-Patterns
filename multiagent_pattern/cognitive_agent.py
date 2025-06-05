from typing import Callable, Dict, List, Any, Optional
from inspect import signature


class CognitiveAgent:
    def __init__(
        self,
        name: str,
        mode: str,
        llm_fn: Callable[[List[Dict[str, str]], List[Dict[str, Any]]], Dict[str, Any]],
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        hooks: Optional[Dict[str, Callable[..., None]]] = None,
        max_steps: int = 5
    ):
        self.name = name
        self.mode = mode.lower()
        self.llm_fn = llm_fn
        self.tools = tools or {}
        self.hooks = hooks or {}
        self.max_steps = max_steps

    def run(self, task: str) -> Dict[str, Any]:
        if self.mode == "tool":
            return self._run_tool_use(task)
        elif self.mode == "react":
            return self._run_react(task)
        elif self.mode == "reflection":
            return self._run_reflection(task)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # === Tool-Use Pattern ===
    def _run_tool_use(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        tool_specs = [self._tool_spec(k, v) for k, v in self.tools.items()]

        for step in range(self.max_steps):
            llm_output = self.llm_fn(messages, tool_specs)
            self._run_hook("after_llm", step, llm_output)

            if "function_call" in llm_output:
                tool_name = llm_output["function_call"]["name"]
                args = llm_output["function_call"].get("arguments", {})
                result = self.tools[tool_name](**args)
                messages.append({"role": "function", "name": tool_name, "content": str(result)})
                self._run_hook("after_tool", step, tool_name, args, result)
            else:
                return {"output": llm_output.get("content", ""), "messages": messages}

        return {"output": messages[-1]["content"], "messages": messages}

    # === ReAct Pattern ===
    def _run_react(self, task: str):
        history = []
        tool_specs = [self._tool_spec(k, v) for k, v in self.tools.items()]
        for step in range(self.max_steps):
            prompt = [{"role": "system", "content": f"You are agent {self.name} using ReAct to solve a task."}]
            prompt.append({"role": "user", "content": f"Task: {task}"})
            messages = prompt + history
            llm_resp = self.llm_fn(messages, tool_specs)

            self._run_hook("after_llm", step, llm_resp)

            if "function_call" in llm_resp:
                tool_name = llm_resp["function_call"]["name"]
                args = llm_resp["function_call"].get("arguments", {})
                result = self.tools[tool_name](**args)
                history.append({"role": "function", "name": tool_name, "content": str(result)})
                self._run_hook("after_tool", step, tool_name, args, result)
            else:
                output = llm_resp.get("content", "")
                history.append({"role": "assistant", "content": output})
                if "final answer" in output.lower():
                    return {"output": output, "history": history}
        return {"output": history[-1]["content"], "history": history}

    # === Reflection Pattern ===
    def _run_reflection(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        initial = self.llm_fn(messages, [])
        messages.append({"role": "assistant", "content": initial["content"]})
        self._run_hook("after_generate", messages, initial)

        reflection_messages = [
            {"role": "system", "content": "Reflect on the quality of this answer and suggest improvements."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial["content"]}
        ]
        reflection = self.llm_fn(reflection_messages, [])
        reflection_messages.append({"role": "assistant", "content": reflection["content"]})
        self._run_hook("after_reflection", reflection_messages, reflection)

        improve_input = reflection_messages + [{"role": "user", "content": "Improve the original based on this."}]
        improved = self.llm_fn(improve_input, [])
        return {
            "initial": initial["content"],
            "reflection": reflection["content"],
            "improved": improved["content"]
        }

    def _tool_spec(self, name: str, fn: Callable) -> Dict[str, Any]:
        sig = signature(fn)
        return {
            "name": name,
            "description": fn.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in sig.parameters},
                "required": list(sig.parameters)
            }
        }

    def _run_hook(self, name, *args, **kwargs):
        if name in self.hooks:
            self.hooks[name](*args, **kwargs)
