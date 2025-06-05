from typing import Callable, Dict, List, Any

class ToolUsePattern:
    def __init__(
        self,
        llm_fn: Callable[[List[Dict[str, str]], List[Dict[str, Any]]], Dict[str, Any]],
        tools: Dict[str, Callable[..., Any]],
        max_steps: int = 3,
        hooks: Dict[str, Callable[..., None]] = None
    ):
        """
        :param llm_fn: Function that takes (messages, tool_specs) and returns an LLM response.
        :param tools: Dict of available tools the model can invoke.
        :param max_steps: Max tool-call loops before stopping.
        :param hooks: Optional hooks like logging.
        """
        self.llm_fn = llm_fn
        self.tools = tools
        self.max_steps = max_steps
        self.hooks = hooks or {}

    def run(self, prompt: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        tool_specs = [self._tool_spec(name, fn) for name, fn in self.tools.items()]

        for step in range(self.max_steps):
            llm_response = self.llm_fn(messages, tool_specs)
            messages.append({"role": "assistant", "content": llm_response.get("content", ""), "function_call": llm_response.get("function_call")})

            self._run_hook("after_llm_response", step, llm_response)

            if "function_call" in llm_response:
                tool_name = llm_response["function_call"]["name"]
                arguments = llm_response["function_call"].get("arguments", {})

                if tool_name not in self.tools:
                    raise ValueError(f"Tool '{tool_name}' not found.")

                tool_result = self.tools[tool_name](**arguments)
                messages.append({
                    "role": "function",
                    "name": tool_name,
                    "content": str(tool_result)
                })

                self._run_hook("after_tool_use", step, tool_name, arguments, tool_result)

            else:
                return {
                    "final_output": llm_response.get("content", ""),
                    "messages": messages
                }

        return {
            "final_output": messages[-1]["content"],
            "messages": messages,
            "warning": "Max steps reached"
        }

    def _tool_spec(self, name: str, fn: Callable[..., Any]) -> Dict[str, Any]:
        return {
            "name": name,
            "description": fn.__doc__ or "",
            "parameters": self._extract_signature(fn)
        }

    def _extract_signature(self, fn: Callable[..., Any]) -> Dict[str, Any]:
        from inspect import signature
        sig = signature(fn)
        return {
            "type": "object",
            "properties": {
                k: {"type": "string"} for k in sig.parameters
            },
            "required": list(sig.parameters.keys())
        }

    def _run_hook(self, hook_name: str, *args, **kwargs):
        if hook_name in self.hooks:
            self.hooks[hook_name](*args, **kwargs)
