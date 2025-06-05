from typing import Callable, Dict, List

class ReflectionPattern:
    def __init__(
        self,
        generate_fn: Callable[[List[Dict[str, str]]], str],
        reflect_fn: Callable[[List[Dict[str, str]]], str] = None,
        improve_fn: Callable[[List[Dict[str, str]]], str] = None,
        hooks: Dict[str, Callable[..., None]] = None,
    ):
        """
        :param generate_fn: Function that accepts a message history and returns output.
        :param reflect_fn: Function that accepts reflection history and returns reflection.
        :param improve_fn: Function that accepts reflection history and returns improved output.
        :param hooks: Optional lifecycle hooks for logging, etc.
        """
        self.generate_fn = generate_fn
        self.reflect_fn = reflect_fn or self.default_reflect
        self.improve_fn = improve_fn or self.default_improve
        self.hooks = hooks or {}

    def run(self, user_prompt: str) -> Dict[str, str]:
        # ===== Generation Phase =====
        generation_history = [{"role": "user", "content": user_prompt}]
        initial = self.generate_fn(generation_history)
        generation_history.append({"role": "assistant", "content": initial})
        self._run_hook("after_generate", generation_history, initial)

        # ===== Reflection Phase =====
        reflection_history = [
            {"role": "system", "content": "You're a critical thinker helping improve responses."},
            {"role": "user", "content": f"Here's the task:\n{user_prompt}"},
            {"role": "assistant", "content": f"Initial response:\n{initial}"},
        ]
        reflection = self.reflect_fn(reflection_history)
        reflection_history.append({"role": "assistant", "content": reflection})
        self._run_hook("after_reflection", reflection_history, reflection)

        # ===== Improvement Phase =====
        improvement_history = reflection_history + [
            {"role": "user", "content": "Now rewrite the initial response using the reflection above."}
        ]
        improved = self.improve_fn(improvement_history)
        reflection_history.append({"role": "assistant", "content": improved})
        self._run_hook("after_improve", reflection_history, improved)

        return {
            "initial": initial,
            "reflection": reflection,
            "improved": improved,
            "gen_history": generation_history,
            "reflect_history": reflection_history,
        }

    def _run_hook(self, hook_name: str, *args, **kwargs):
        if hook_name in self.hooks:
            self.hooks[hook_name](*args, **kwargs)

    @staticmethod
    def default_reflect(history: List[Dict[str, str]]) -> str:
        return "Reflection placeholder – you should override this."

    @staticmethod
    def default_improve(history: List[Dict[str, str]]) -> str:
        return "Improved output placeholder – you should override this."
