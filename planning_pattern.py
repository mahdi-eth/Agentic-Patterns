from typing import Callable, Dict, List, Any

class PlanningPattern:
    def __init__(
        self,
        llm_fn: Callable[[List[Dict[str, str]], List[Dict[str, Any]]], Dict[str, Any]],
        tools: Dict[str, Callable[..., Any]],
        max_steps: int = 5,
        hooks: Dict[str, Callable[..., None]] = None,
        react_prompt_builder: Callable[[str, List[Dict[str, str]]], List[Dict[str, str]]] = None
    ):
        """
        :param llm_fn: Function that takes (messages, tool_specs) and returns LLM output.
        :param tools: Dict of action tools (like search, calc, etc).
        :param max_steps: Maximum ReAct loop iterations.
        :param hooks: Optional lifecycle hooks.
        :param react_prompt_builder: Builds full prompt context for ReAct reasoning.
        """
        self.llm_fn = llm_fn
        self.tools = tools
        self.max_steps = max_steps
        self.hooks = hooks or {}
        self.react_prompt_builder = react_prompt_builder or self.default_prompt_builder

    def run(self, task: str) -> Dict[str, Any]:
        history = []
        tool_specs = [self._tool_spec(name, fn) for name, fn in self.tools.items()]

        for step in range(self.max_steps):
            prompt = self.react_prompt_builder(task, history)
            llm_resp = self.llm_fn(prompt, tool_specs)

            self._run_hook("after_llm", step, llm_resp)

            if "function_call" in llm_resp:
                tool_name = llm_resp["function_call"]["name"]
                arguments = llm_resp["function_call"].get("arguments", {})
                if tool_name not in self.tools:
                    raise ValueError(f"Unknown tool: {tool_name}")

                result = self.tools[tool_name](**arguments)
                observation = f"Observation: {result}"
                history.append({"role": "function", "name": tool_name, "content": result})

                self._run_hook("after_action", step, tool_name, arguments, result)
            else:
                output = llm_resp.get("content", "")
                if self._is_final_answer(output):
                    return {
                        "final_output": output,
                        "history": history
                    }

                # Model just thought, not acted
                history.append({"role": "assistant", "content": output})

        return {
            "final_output": history[-1]["content"] if history else "",
            "history": history,
            "warning": "Max steps reached"
        }

    def _is_final_answer(self, text: str) -> bool:
        return "Final Answer:" in text or text.lower().strip().startswith("the answer is")

    def _tool_spec(self, name: str, fn: Callable[..., Any]) -> Dict[str, Any]:
        from inspect import signature
        sig = signature(fn)
        return {
            "name": name,
            "description": fn.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {
                    k: {"type": "string"} for k in sig.parameters
                },
                "required": list(sig.parameters.keys())
            }
        }

    def default_prompt_builder(self, task: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        base = [{"role": "system", "content": "You are a ReAct-style agent that thinks step by step and takes actions when needed."}]
        base.append({"role": "user", "content": f"Task: {task}"})
        return base + history

    def _run_hook(self, name: str, *args, **kwargs):
        if name in self.hooks:
            self.hooks[name](*args, **kwargs)
